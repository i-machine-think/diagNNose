import torch

from .shapley_tensor import ShapleyTensor


class GCDTensor(ShapleyTensor):
    def mul_contributions(self, *args, **kwargs):
        gate, source = args

        if not isinstance(gate, ShapleyTensor):
            contributions = [
                torch.mul(gate, contribution, **kwargs)
                for contribution in source.contributions
            ]
        elif not isinstance(source, ShapleyTensor):
            contributions = [
                torch.mul(contribution, source, **kwargs)
                for contribution in gate.contributions
            ]
        else:
            gate_contributions = sum(gate.contributions)
            contributions = [
                torch.mul(gate_contributions, source_contribution, **kwargs)
                for source_contribution in source.contributions
            ]

        return contributions
