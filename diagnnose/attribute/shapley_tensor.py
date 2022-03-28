from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
from warnings import warn

import torch
from torch import Tensor

from . import utils


class ShapleyTensor:
    """A ShapleyTensor wraps a torch Tensor. It allows the tensor to
    be decomposed into a sum of tensors, that each define the
    contribution of a feature to the tensor.

    ShapleyTensors can be passed to any type of torch model. For each
    operation in the model the intermediate Shapley values are
    calculated for the list of contributions. This is done using
    `__torch_function__`, that allows to override tensor operations.

    Parameters
    ----------
    data : Tensor
        Input tensor that is decomposed into a sum of contributions.
    contributions : List[Tensor]
        List of contributions that should sum up to `data`.
    shapley_factors : List[Tuple[List[int], int]], optional
        Shapley factors that are calculated with `calc_shapley_factors`.
        To prevent unnecessary compute these factors are passed on to
        subsequent ShapleyTensors.
    num_samples : int, optional
        Number of feature permutation samples. Increasing the number of
        samples will reduce the variance of the approximation. If not
        provided the exact Shapley values will be computed.
    validate : bool, optional
        Toggle to validate at each step whether `contributions` still
        sums up to `data`. Defaults to False.
    baseline_partition : int, optional
        Index of the contribution partition to which the baseline fn(0)
        will be added. If we do not add this baseline the contributions
        won't sum up to the full output. Defaults to 0.
    """

    def __init__(
        self,
        data: Tensor,
        contributions: Optional[List[Tensor]] = None,
        shapley_factors: Optional[List[Tuple[List[int], int]]] = None,
        num_samples: Optional[int] = None,
        validate: bool = False,
        baseline_partition: int = 0,
    ):
        if not utils.MONKEY_PATCH_PERFORMED:
            utils.monkey_patch()
            utils.MONKEY_PATCH_PERFORMED = True

        self.data = data
        self.contributions = contributions or []
        self.shapley_factors = shapley_factors
        self.num_samples = num_samples
        self.validate = validate
        self.baseline_partition = baseline_partition

        self.current_fn: Optional[str] = None
        self.new_data: Optional[Union[Tensor, Iterable[Tensor]]] = None

        if (
            len(self.contributions) > 0
            and shapley_factors is None
            and num_samples is None
        ):
            self.shapley_factors = utils.calc_shapley_factors(self.num_features)

    def __torch_function__(self, fn, _types, args=(), kwargs=None):
        self.current_fn = fn.__name__

        kwargs = kwargs or {}

        self.new_data = fn(*map(utils.unwrap, args), **kwargs)

        new_contributions = self._calc_contributions(fn, *args, **kwargs)

        return self._pack_output(self.new_data, new_contributions)

    @property
    def num_features(self) -> int:
        return len(self.contributions)

    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    def dim(self, *args, **kwargs):
        return self.data.dim(*args, **kwargs)

    def __iter__(self):
        """Allows a ShapleyTensor to be unpacked directly as:

        .. code-block:: python

            data, contributions = shapley_tensor
        """
        yield from [self.data, self.contributions]

    def __len__(self):
        return len(self.data)

    def __getattr__(self, item: str) -> Any:
        """
        Handles torch methods that are called on a tensor itself, like
        ``tensor.add(*args)`` or ``tensor.view(*args)``.
        """
        attr = getattr(self.data, item)

        if isinstance(attr, Callable):

            def attr_wrapper(*args, **kwargs):
                torch_fn = getattr(torch, attr.__name__, None)

                if isinstance(torch_fn, Callable):
                    # Captures function calls like tensor.transpose(*args), and returns them
                    # like torch.transpose(tensor, *args), so the function call can be captured
                    # using __torch_function__.
                    if attr.__name__ == "reshape":
                        return torch_fn(self, args, **kwargs)

                    return torch_fn(self, *args, **kwargs)
                else:
                    # Captures tensor functions that don't exist as a stand-alone torch method,
                    # such as tensor.view(*args). Applies the same function to all contributions.
                    output = attr(*args, **kwargs)
                    contributions = [
                        getattr(contribution, item)(*args, **kwargs)
                        for contribution in self.contributions
                    ]

                    return self._pack_output(output, contributions)

            return attr_wrapper
        else:
            return attr

    def __getitem__(self, index):
        if isinstance(index, ShapleyTensor):
            data = self.data[index.data]
            contributions = [self.data[c] for c in index.contributions]
            self.validate = self.validate or index.validate
        else:
            data = self.data[index]
            contributions = [contribution[index] for contribution in self.contributions]

        # We return type(self) to allow a subclass that derives from ShapleyTensor to be preserved.
        tensor_type = type(self)

        return tensor_type(
            data,
            contributions=contributions,
            shapley_factors=self.shapley_factors,
            num_samples=self.num_samples,
            validate=self.validate,
            baseline_partition=self.baseline_partition,
        )

    def __setitem__(self, index, value):
        self.data[index] = value.data

        # We pad the current contributions if the value that is set contains more contributions
        # than the current ShapleyTensor.
        if len(self.contributions) < len(value.contributions):
            extra_contributions = len(value.contributions) - len(self.contributions)
            for _ in range(extra_contributions):
                self.contributions.append(torch.zeros_like(self.data))

        for c_idx, contribution in enumerate(self.contributions):
            contribution[index] = value.contributions[c_idx]

    def _validate_contributions(self, data, contributions) -> None:
        """ Asserts whether the contributions sum up to the full tensor. """
        diff = (data - sum(contributions)).float()
        mean_diff = torch.mean(diff)
        max_diff = torch.max(torch.abs(diff))
        if not torch.allclose(data, sum(contributions), rtol=1e-3, atol=1e-3):
            warn(
                f"Contributions don't sum up to the provided tensor, with a mean difference of "
                f"{mean_diff:.3E} and a max difference of {max_diff:.3E}. "
                f"Current function is: {self.current_fn}"
            )

    def _pack_output(
        self,
        new_data: Union[Tensor, Iterable[Tensor]],
        new_contributions: Union[List[Tensor], Iterable[List[Tensor]]],
    ) -> Any:
        """Packs the output and its corresponding contributions into a
        new ShapleyTensor.

        If the output is an iterable (e.g. with a .split operation) the
        type structure of the output is preserved.
        """
        if isinstance(new_data, torch.Tensor):
            tensor_type = type(self)

            if self.validate and len(new_contributions) > 0:
                self._validate_contributions(new_data, new_contributions)

            return tensor_type(
                new_data,
                contributions=new_contributions,
                shapley_factors=self.shapley_factors,
                num_samples=self.num_samples,
                validate=self.validate,
                baseline_partition=self.baseline_partition,
            )
        elif self.current_fn == "_pack_padded_sequence":
            new_contributions = [c[0] for c in new_contributions]

            return self._pack_output(new_data[0], new_contributions), new_data[1]
        elif isinstance(new_data, (list, tuple)):
            iterable_type = type(new_data)

            if self.num_features == 0:
                return iterable_type(self._pack_output(item, []) for item in new_data)

            return iterable_type(
                self._pack_output(data, contributions)
                for data, contributions in zip(new_data, new_contributions)
            )

        return new_data

    def _calc_contributions(
        self, fn, *args, **kwargs
    ) -> Union[List[Tensor], Sequence[List[Tensor]]]:
        """
        Some methods have custom behaviour for how the output is
        decomposed into a new set of contributions.
        """
        if self.num_features == 0:
            return []
        elif hasattr(self, f"{fn.__name__}_contributions"):
            fn_contributions = getattr(self, f"{fn.__name__}_contributions")
            return fn_contributions(*args, **kwargs)
        elif fn.__name__ in [
            "squeeze",
            "unsqueeze",
            "index_select",
            "_pack_padded_sequence",
        ]:
            old_contributions = args[0].contributions
            return [fn(c, *args[1:], **kwargs) for c in old_contributions]

        return self._calc_shapley_contributions(fn, *args, **kwargs)

    def _calc_shapley_contributions(
        self, fn, *args, **kwargs
    ) -> Union[List[Tensor], Sequence[List[Tensor]]]:
        """ Calculates the Shapley decomposition of the current fn. """
        warning_msg = f"Current operation {self.current_fn} is not supported for Shapley calculation"
        if isinstance(self.new_data, Iterable):
            assert all(isinstance(data, Tensor) for data in self.new_data), warning_msg
        else:
            assert isinstance(self.new_data, Tensor), warning_msg

        if self.num_samples is None:
            return utils.calc_exact_shapley_values(
                fn,
                self.num_features,
                self.shapley_factors,
                self.new_data,
                self.baseline_partition,
                *args,
                **kwargs,
            )
        else:
            return utils.calc_sample_shapley_values(
                fn,
                self.num_features,
                self.num_samples,
                self.new_data,
                self.baseline_partition,
                *args,
                **kwargs,
            )

    def cat_contributions(self, *args, **kwargs):
        def _pad_contributions(arg: Union[Tensor, ShapleyTensor]):
            """A non-ShapleyTensor only contributes to the baseline
            partition, and is padded with 0s.
            """
            if isinstance(arg, ShapleyTensor):
                return arg.contributions
            elif isinstance(arg, Tensor):
                padded_contribution = self.num_features * [torch.zeros_like(arg)]
                padded_contribution[self.baseline_partition] = arg

                return padded_contribution
            else:
                raise TypeError

        padded_contributions = [_pad_contributions(tensor) for tensor in args[0]]

        contributions = [
            torch.cat(
                [contribution[idx] for contribution in padded_contributions], **kwargs
            )
            for idx in range(self.num_features)
        ]

        return contributions

    @staticmethod
    def split_contributions(*args, **kwargs):
        shapley_tensor, split_size_or_sections = args

        raw_splits = [
            torch.split(contribution, split_size_or_sections, **kwargs)
            for contribution in shapley_tensor.contributions
        ]

        num_splits = len(raw_splits[0])

        contributions = [
            [split[idx] for split in raw_splits] for idx in range(num_splits)
        ]

        return contributions

    def dropout_contributions(self, *args, **kwargs):
        """In principle dropout should be disabled when calculating
        Shapley contributions, but we should still take care of it.

        We determine the dropout mask by looking at the difference
        between the new output data and the input.
        """
        dropout_mask = self.new_data != args[0].data

        contributions = args[0].contributions
        for contribution in contributions:
            contribution[dropout_mask] = 0.0

        return contributions

    def dropout2d_contributions(self, *args, **kwargs):
        return self.dropout_contributions(*args, **kwargs)

    def dropout3d_contributions(self, *args, **kwargs):
        return self.dropout_contributions(*args, **kwargs)

    def add_contributions(self, *args, **kwargs):
        """ Non-ShapleyTensors are added to the baseline partition. """
        input_, other = args

        if not isinstance(input_, ShapleyTensor):
            contributions = other.contributions
            contributions[self.baseline_partition] += input_
        elif not isinstance(other, ShapleyTensor):
            contributions = input_.contributions
            contributions[self.baseline_partition] += other
        else:
            contributions = [
                torch.add(con1, con2, **kwargs)
                for con1, con2 in zip(input_.contributions, other.contributions)
            ]

        return contributions

    def mul_contributions(self, *args, **kwargs):
        input_, other = args
        if isinstance(input_, torch.Tensor):
            contributions = [
                torch.mul(input_, contribution, **kwargs)
                for contribution in other.contributions
            ]
        elif isinstance(other, torch.Tensor):
            contributions = [
                torch.mul(contribution, other, **kwargs)
                for contribution in input_.contributions
            ]
        else:
            contributions = self._calc_shapley_contributions(torch.mul, *args, **kwargs)

        return contributions

    def linear_contributions(self, *args, **kwargs):
        input_, weight = args
        bias = kwargs.get("bias", None)

        output = self.matmul_contributions(input_, weight.t())

        # Bias term is added to the baseline partition
        if bias is not None:
            output[self.baseline_partition] += bias

        return output

    def matmul_contributions(self, *args, **kwargs):
        input_, other = args
        if isinstance(input_, torch.Tensor):
            contributions = [
                torch.matmul(input_, contribution, **kwargs)
                for contribution in other.contributions
            ]
        elif isinstance(other, torch.Tensor):
            contributions = [
                torch.matmul(contribution, other, **kwargs)
                for contribution in input_.contributions
            ]
        else:
            contributions = self._calc_shapley_contributions(
                torch.matmul, *args, **kwargs
            )

        return contributions

    def __add__(self, other):
        return torch.add(self, other)

    def __radd__(self, other):
        return torch.add(other, self)

    def __sub__(self, other):
        return torch.sub(self, other)

    def __rsub__(self, other):
        return torch.sub(other, self)

    def __matmul__(self, other):
        return torch.matmul(self, other)

    def __rmatmul__(self, other):
        return torch.matmul(other, self)

    def __mul__(self, other):
        return torch.mul(self, other)

    def __rmul__(self, other):
        return torch.mul(other, self)

    def __pow__(self, other):
        return torch.pow(self, other)

    def __div__(self, other):
        return torch.div(self, other)

    def __rdiv__(self, other):
        return torch.div(other, self)

    def __mod__(self, other):
        return torch.fmod(self, other)

    def __truediv__(self, other):
        return torch.true_divide(self, other)

    def __floordiv__(self, other):
        return torch.div(self, other).floor()

    def __rfloordiv__(self, other):
        return torch.div(other, self).floor()

    def __abs__(self):
        return torch.abs(self)

    def __and__(self, other):
        return torch.logical_and(self, other)

    def __ge__(self, other):
        return torch.ge(self, other)

    def __gt__(self, other):
        return torch.gt(self, other)

    def __invert__(self):
        return torch.logical_not(self)

    def __le__(self, other):
        return torch.le(self, other)

    def __lt__(self, other):
        return torch.lt(self, other)

    def __ne__(self, other):
        return torch.ne(self, other)

    def __neg__(self):
        return torch.neg(self)

    def __or__(self, other):
        return torch.logical_or(self, other)

    def __xor__(self, other):
        return torch.logical_xor(self, other)
