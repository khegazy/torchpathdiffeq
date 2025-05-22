from typing import Any
import torch
from dataclasses import dataclass

from .base import steps

@dataclass
class _Tableau():
    c: torch.Tensor
    b: torch.Tensor
    b_error: torch.Tensor
    
    def to_dtype(self, dtype):
        self.c = self.c.to(dtype)
        self.b = self.b.to(dtype)
        self.b_error = self.b_error.to(dtype)
    
    def to_device(self, device):
        self.c = self.c.to(device) 
        self.b = self.b.to(device) 
        self.b_error = self.b_error.to(device)


class MethodClass():
    order: int
    tableau: _Tableau

    def __init__(self, order, tableau):
        self.order = order
        self.tableau = tableau
    
    def to_dtype(self, dtype):
        self.tableau.to_dtype(dtype)
    
    def to_device(self, device):
        self.tableau.to_device(device)



_ADAPTIVE_HEUN = MethodClass(
    order = 2,
    tableau = _Tableau(
        c = torch.tensor([0.0, 1.0], dtype=torch.float64),
        b = torch.tensor([[0.5, 0.5]], dtype=torch.float64),
        b_error = torch.tensor([[0.5, -0.5]], dtype=torch.float64)
    )
)

_FEHLBERG2 = MethodClass(
    order = 2,
    tableau = _Tableau(
        c = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        b = torch.tensor([1 / 512, 255 / 256, 1 / 512], dtype=torch.float64),
        b_error = torch.tensor([-1 / 512, 0, 1 / 512], dtype=torch.float64)
    )
)

_BOGACKI_SHAMPINE = MethodClass(
    order = 3,
    tableau = _Tableau(
        c = torch.tensor([0.0, 0.5, 0.75, 1.0], dtype=torch.float64),
        b = torch.tensor([2 / 9, 1 / 3, 4 / 9, 0.], dtype=torch.float64),
        b_error = torch.tensor(
            [2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, -1 / 8],
            dtype=torch.float64
        )
    )
)

_DORMAND_PRINCE_SHAMPINE = MethodClass(
    order = 5,
    tableau = _Tableau(
        c = torch.tensor(
            [0.0, 1.0/5, 3.0/10, 4.0/5, 8.0/9, 1., 1.], dtype=torch.float64
        ),
        b = torch.tensor(
            [
                35 / 384,
                0,
                500 / 1113,
                125 / 192,
                -2187 / 6784,
                11 / 84,
                0
            ],
            dtype=torch.float64
        ),
        b_error = torch.tensor(
            [
                35 / 384 - 1951 / 21600,
                0,
                500 / 1113 - 22642 / 50085,
                125 / 192 - 451 / 720,
                -2187 / 6784 - -12231 / 42400,
                11 / 84 - 649 / 6300,
                -1. / 60.,
            ],
            dtype=torch.float64
        )
    )
)


UNIFORM_METHODS = {
    'adaptive_heun' : _ADAPTIVE_HEUN,
    'fehlberg2' : _FEHLBERG2,
    'bosh3' : _BOGACKI_SHAMPINE,
    'dopri5' : _DORMAND_PRINCE_SHAMPINE
}


###############################################
#####  Variable Sampling Adaptive Methods #####
###############################################

class _VariableSubclass():
    def __init__(self, device):
        self.device = device
    
    def to_device(self, device):
        raise NotImplementedError
    
    def to_dtype(self, dtype):
        raise NotImplementedError

class _VARIABLE_SECOND_ORDER(_VariableSubclass):
    order = 2
    n_tableau_c = 2
    
    def __init__(self, device=None) -> None:
        super().__init__(device)
        self.device = device
        self.tableau = _ADAPTIVE_HEUN.tableau
    
    def to_device(self, device):
        self.tableau.to_device(device)
    
    def to_dtype(self, dtype):
        self.tableau.to(dtype)

    def tableau_b(self, c):
        b = self.tableau.b
        b_error = self.tableau.b_error
        return b, b_error


class _VARIABLE_THIRD_ORDER(_VariableSubclass):
    order = 3
    n_tableau_c = 3
    
    def __init__(self, device=None) -> None:
        super().__init__(device)
        self.device = device
        self.b_delta = torch.tensor(
            [[0.5, 0.0, 0.5]], dtype=torch.float64, device=self.device
        )
    
    def to_device(self, device):
        self.b_delta = self.b_delta.to(device)
    
    def to_dtype(self, dtype):
        self.b_delta = self.b_delta.to(dtype)
    
    def _b0(self, a):
        return 0.5 - 1./(6*a)
    
    def _ba(self, a):
        return 1./(6*a*(1 - a))

    def _b1(self, a):
        return (2. - 3*a)/(6*(1. - a))
    
    def tableau_b(self, c):
        """
        Generic third order method by Sanderse and Veldman
              degr=P1                 degr=P

        c |      b               c |      b
        ------------------       ------------------
        0 | 1/2 - 1/(6a)         0 | 1/2 - 1/(6a)
        a | 1/(6a(1-a))          a | 1/(6a(1-a))
        1 | (2-3a)/(6(1-a))      z | 0
                                 1 | (2-3a)/(6(1-a))   
        
        c: [n, p or p1, d]
        """

        a = c[:,1,0]
        b = torch.stack([self._b0(a), self._ba(a), self._b1(a)]).transpose(0,1)
        b_error = b - self.b_delta
        
        return b, b_error

VARIABLE_METHODS = {
    'adaptive_heun' : _VARIABLE_SECOND_ORDER,
    'generic3' : _VARIABLE_THIRD_ORDER
}


def _get_method(sampling_type, method_name, device, dtype):
    if sampling_type == steps.ADAPTIVE_UNIFORM:
        method = UNIFORM_METHODS[method_name]
        #method.tableau = _tableau_to_device(method.tableau, device) 
    else:
        method = VARIABLE_METHODS[method_name](device)

    method.to_device(device)
    method.to_dtype(dtype) 
    return method
