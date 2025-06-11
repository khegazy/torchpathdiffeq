import torch
from torch import nn

class AdjointClass(torch.autograd.Function):
    def forward():
    
    def backward():
    
    def adjoint_augmented():
    
    def setup_adjoint():
        raise NotImplementedError

def basic_usage_fxn(method_name, fxn, t_init=0, t_final=1):
    integrator = get_parallel_RK_solver(method_name, fxn, t_init=t_init, t_final=t_final)
    ans = AdjointClass.apply(integrator)

