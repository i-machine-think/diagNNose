from typing import Callable, List
from warnings import warn

from torch import Tensor

from . import utils


class ShapleyTensor:
    """ A ShapleyTensor wraps a torch Tensor. It allows the tensor to
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
    validate : bool, optional
        Toggle to validate at each step whether `contributions` still
        sums up to `data`. Defaults to False.
    """

    if not utils.MONKEY_PATCH_PERFORMED:
        utils.monkey_patch()
        utils.MONKEY_PATCH_PERFORMED = True

    def __init__(
        self,
        data: Tensor,
        contributions: Optional[List[Tensor]] = None,
        shapley_factors: Optional[List[Tuple[List[int], int]]] = None,
        validate: bool = False,
    ):
        self.data = data
        self.contributions = contributions or []
        self.shapley_factors = shapley_factors
        self.validate = validate

        self.current_fn: Optional[str] = None

        if len(self.contributions) > 0:
            if validate:
                self.validate_contributions()

            if shapley_factors is None:
                self.shapley_factors = utils.calc_shapley_factors(
                    len(contributions) - 1
                )

    def __torch_function__(self, fn, _types, args=(), kwargs=None):
        self.current_fn = fn.__name__

        kwargs = kwargs or {}

        data = fn(*map(utils.unwrap, args), **kwargs)
        if len(self.contributions) == 0:
            contributions = []
        else:
            contributions = self.calc_contributions(fn, *args, **kwargs)

        output = self.transform_output(data, contributions)

        return output

    def calc_contributions(self, fn, *args, **kwargs) -> List[Tensor]:
        if hasattr(self, f"{fn.__name__}_contributions"):
            fn = getattr(self, f"{fn.__name__}_contributions")
            return fn(*args, **kwargs)
        elif fn.__name__ in [
            "squeeze",
            "unsqueeze",
            "index_select",
            "_pack_padded_sequence",
        ]:
            old_contributions = args[0].contributions
            return [fn(c, *args[1:], **kwargs) for c in old_contributions]

        return self.calc_shapley_contributions(fn, *args, **kwargs)

    def calc_shapley_contributions(self, fn, *args, **kwargs) -> List[Tensor]:
        contributions = []

        for f_idx in range(self.num_features):
            other_ids = torch.tensor(
                [i for i in range(self.num_features) if i != f_idx]
            )

            contribution = 0.0

            for coalition_ids, factor in self.shapley_factors:
                coalition = list(other_ids[coalition_ids])
                args_wo = [
                    unwrap(arg, attr="contributions", coalition=coalition)
                    for arg in args
                ]

                coalition += [f_idx]
                args_with = [
                    unwrap(arg, attr="contributions", coalition=coalition)
                    for arg in args
                ]

                contribution += factor * (
                    fn(*args_with, **kwargs) - fn(*args_wo, **kwargs)
                )

            contribution /= factorial(self.num_features)
            contributions.append(contribution)

        # Add baseline to default feature ([0]).
        zero_input_args = [
            unwrap(arg, attr="contributions", coalition=[]) for arg in args
        ]
        contributions[0] += fn(*zero_input_args, **kwargs)

        return contributions

    def transform_output(
        self, output: Any, contributions: Optional[List[Tensor]] = None
    ) -> Any:
        if isinstance(output, torch.Tensor):
            return ShapleyTensor(
                output,
                contributions=contributions,
                shapley_factors=self.shapley_factors,
                validate=self.validate,
            )
        elif self.current_fn == "_pack_padded_sequence":
            contributions = [c[0] for c in contributions]
            return self.transform_output(output[0], contributions), output[1]
        elif isinstance(output, (list, tuple)):
            iterable_type = type(output)

            if len(contributions) == 0:
                return iterable_type(self.transform_output(item) for item in output)

            return iterable_type(
                self.transform_output(item, contributions[idx])
                for idx, item in enumerate(output)
            )

        return output

    def validate_contributions(self) -> None:
        """ Asserts whether the contributions sum up to the full tensor. """
        diff = (self.data - sum(self.contributions)).float()
        mean_diff = torch.mean(diff)
        max_diff = torch.max(torch.abs(diff))
        if not torch.allclose(self.data, sum(self.contributions), rtol=1e-3, atol=1e-3):
            warn(
                f"Contributions don't sum up to the provided tensor, with a mean difference of "
                f"{mean_diff:.3E} and a max difference of {max_diff:.3E}."
            )

    def __len__(self):
        return len(self.data)

    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    def __iter__(self):
        """ Allows a ShapleyTensor to be unpacked directly. """
        yield from [self.data, self.contributions]

    def __getattr__(self, item: str) -> Any:
        attr = getattr(self.data, item)

        if isinstance(attr, Callable):

            def attr_wrapper(*args, **kwargs):
                if hasattr(torch, attr.__name__) and isinstance(
                    getattr(torch, attr.__name__), Callable
                ):
                    torch_fn = getattr(torch, attr.__name__)

                    if attr.__name__ == "reshape":
                        return torch_fn(self, args, **kwargs)

                    return torch_fn(self, *args, **kwargs)
                else:
                    output = attr(*args, **kwargs)
                    contributions = [
                        getattr(contribution, item)(*args, **kwargs)
                        for contribution in self.contributions
                    ]

                    return self.transform_output(output, contributions)

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

        return ShapleyTensor(
            data,
            contributions=contributions,
            shapley_factors=self.shapley_factors,
            validate=self.validate,
        )

    def __setitem__(self, index, value):
        self.data[index] = value.data

        # We pad the current contributions if the value that is set contains more contributions
        # than the current ShapleyTensor.
        if len(self.contributions) < len(value.contributions):
            extra_contributions = len(value.contributions) - len(self.contributions)
            self.contributions.append(extra_contributions * torch.zeros_like(self.data))

        for c_idx, contribution in enumerate(self.contributions):
            contribution[index] = value.contributions[c_idx]

    def cat_contributions(self, *args, **kwargs):
        # A non-ShapleyTensor only contributes to the default feature, and is padded with 0s.
        all_contributions = [
            arg.contributions
            if hasattr(arg, "contributions")
            else [arg] + (self.num_features - 1) * [torch.zeros_like(arg)]
            for arg in args[0]
        ]

        contributions = [
            torch.cat(
                [contribution[idx] for contribution in all_contributions], **kwargs
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

    @staticmethod
    def add_contributions(*args, **kwargs):
        arg1, arg2 = args
        if not isinstance(arg1, ShapleyTensor):
            contributions = [
                torch.add(arg1, arg2.contributions[0], **kwargs),
                *(
                    torch.add(torch.zeros_like(arg), arg, **kwargs)
                    for arg in arg2.contributions[1:]
                ),
            ]
        elif not isinstance(arg2, ShapleyTensor):
            contributions = [
                torch.add(arg1.contributions[0], arg2, **kwargs),
                *arg1.contributions[1:],
            ]
        else:
            contributions = [
                torch.add(con1, con2, **kwargs)
                for con1, con2 in zip(arg1.contributions, arg2.contributions)
            ]

        return contributions

    def matmul_contributions(self, *args, **kwargs):
        arg1, arg2 = args
        if isinstance(arg1, torch.Tensor):
            contributions = [
                torch.matmul(arg1, contribution, **kwargs)
                for contribution in arg2.contributions
            ]
        elif isinstance(arg2, torch.Tensor):
            contributions = [
                torch.matmul(contribution, arg2, **kwargs)
                for contribution in arg1.contributions
            ]
        else:
            contributions = self.calc_shapley_contributions(
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
