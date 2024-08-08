from typing import Any
import torch
from .solvers import degree

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
    
    def __call__(self, dt, degr):
        """
        Generic third order method by Sanderse and Veldman
              degr=P1                 degr=P

        c |      b               c |      b
        ------------------       ------------------
        0 | 1/2 - 1/(6a)         0 | 1/2 - 1/(6a)
        a | 1/(6a(1-a))          a | 1/(6a(1-a))
        1 | (2-3a)/(6(1-a))      z | 0
                                 1 | (2-3a)/(6(1-a))   
        """

        if degr == degree.P:
            mask = torch.arange(len(dt)) % 3 == 0
            a = dt[mask]/h[:]

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
            mask = torch.arange(len(dt)) % 2 == 0
            a = dt[mask]/h[:]
            
            b = torch.concatenate(
                [self._b0(a), self._ba(a), self._b1(a)], dim=1
            )
        
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


   