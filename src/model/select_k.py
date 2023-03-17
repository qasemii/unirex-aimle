import torch
from torch import Tensor
from typing import Callable


# differentiable select k model
class SelectK(torch.nn.Module):
    def __init__(self,
                 diff_fun):
        super().__init__()
        self.diff_fun = diff_fun

    def forward(self, attrs: Tensor) -> Tensor:
        return self.diff_fun(attrs)



