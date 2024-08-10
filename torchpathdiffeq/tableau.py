import torch
import numpy as np
from typing import Any

from .methods import UNIFORM_METHODS, VARIABLE_METHODS
from .solvers import steps


class euler_tableau_b():
    def __init__(self, device=None):
        self.device = device
    

    def __call__(self, c, degr):
        """
        c | b
        -----
        0 | 1
        1 | 0
        """

        return torch.concatenate(
            [torch.ones((1, 1)), torch.zeros((1, 1))],
            dim=-1,
        )

class heun_tableau_b():
    def __init__(self, device=None):
        self.device=device

    def __call__(self, c):
        """
        Heun's Method, aka Trapazoidal Rule
        degr=P1             degr=P

        c |  b              c |  b
        -------             -------
        0 | 0.5             0 | 0.5
        1 | 0.5             a | 0
                            1 | 0.5
        """
        
        b = torch.tensor([[0.5, 0.5]])
        b_error = torch.tensor([[0.5, -0.5]])
        
        return b, b_error


class bogacki_shampine_tableau_b():
    def __init__(self, device=None):
        self.device=device

    def __call__(self, c):
        b = torch.tensor([2 / 9, 1 / 3, 4 / 9, 0.], dtype=torch.float64)
        b_error = torch.tensor([2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, -1 / 8], dtype=torch.float64)
        return b, b_error


class fehlberg2_tableau_b():
    def __init__(self, device=None):
        self.device=device

    def __call__(self, c):
        b = torch.tensor([1 / 512, 255 / 256, 1 / 512], dtype=torch.float64)
        b_error = torch.tensor([-1 / 512, 0, 1 / 512], dtype=torch.float64)
        return b, b_error


class dormand_prince_shampine_tableau_b():
    def __init__(self, device=None):
        self.device=device

    def __call__(self, c):
        b = torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=torch.float64)
        b_error = torch.tensor([
            35 / 384 - 1951 / 21600,
            0,
            500 / 1113 - 22642 / 50085,
            125 / 192 - 451 / 720,
            -2187 / 6784 - -12231 / 42400,
            11 / 84 - 649 / 6300,
            -1. / 60.,
        ], dtype=torch.float64)
        return b, b_error


class _heun_tableau_b():
    def __init__(self, device=None):
        self.device=device

    def __call__(self, c, degr):
        """
        Heun's Method, aka Trapazoidal Rule
        degr=P1             degr=P

        c |  b              c |  b
        -------             -------
        0 | 0.5             0 | 0.5
        1 | 0.5             a | 0
                            1 | 0.5
        """
        
        if degr == degree.P:
            b = torch.ones((len(c), 3))*0.5
            b[:,1] = 0
        else:
            b = torch.ones((1, 2))*0.5
        
        return b


class generic3_tableau_b():
    def __init__(self, device=None) -> None:
        self.device = device
    
    def _b0(self, a):
        return 0.5 - 1./(6*a)
    
    def _ba(self, a):
        return 1./(6*a*(1 - a))

    def _b1(self, a):
        return (2. - 3*a)/(6*(1. - a))
    
    def __call__(self, c, degr):
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
        if degr == degree.P:
            b = torch.concatenate(
                [
                    self._b0(a),
                    self._ba(a),
                    torch.zeros((len(a), 1)),
                    self._b1(a)
                ],
                dim=1
            )
        else:
            b = torch.stack(
                [self._b0(a), self._ba(a), self._b1(a)]
            ).transpose(0,1)
        
        return b


class threeEigth_tableau_b():
    def __init__(self, device=None) -> None:
        self.device = device
    
    def __call__(self, dt, h, degr):
        """
        Generic third order method by Sanderse and Veldman
           degr=P1            degr=P

          c  |  b            c  |  b
        -----------       ------------
        0.0  | 1/8         0.0  | 1/8
        0.33 | 3/8         0.25 | 3/8
        0.66 | 3/8         0.50 | 0
        1.0  | 1/8         0.75 | 3/8
                           1.0  | 1/8  
        """

        if degr == degree.P:
            b = torch.tensor([1./8, 3./8, 3./8, 1./8])
        else:
            b = torch.tensor([1./8, 3./8, 0, 3./8, 1./8])
        
        return b


"""
class TableauUniform():
    def __init__(self, method):
        self.method = method
        self.tableau = UNIFORM_METHODS[self.method].tableau
    
    def get_tableau_b(self):
        return self.tableau.b[None,:,None], self.tableau.b_error[None,:,None]

class TableauVariable():
    def __init__(self, method):
        self.method = method
        self.tableau = VARIABLE_METHODS[self.method].tableau
    
    def calculate_tableau_b(self, t):
        norm_dt = t - t[:,0,None]
        norm_dt = norm_dt/norm_dt[:,-1,None]
        b, b_error = self.tableau_b(norm_dt)
        return b.unsqueeze(-1), b_error.unsqueeze(-1)

"""

  