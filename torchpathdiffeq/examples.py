import torch

class Unity():
    __name__ = "Unity"
    def __init__(self) -> None:
        pass
    
    def __call__(self, t, y=None):
        print("T", t.shape)
        return torch.ones_like(t, device=t.device, dtype=t.dtype)
    
    def solve(self, t_init, t_final):
        return t_final - t_init

class Identity():
    __name__ = "Identity"
    def __init__(self) -> None:
        pass

    def __call__(self, t, y=None):
        return t

    def solve(self, t_init, t_final):
        return 0.5*(t_final**2 - t_init**2)

class Squared:
    __name__ = "Squared"
    def __init__(self) -> None:
        pass
        
    def __call__(self, t, y=None):
        return t**2

    def solve(self, t_init, t_final):
        return (t_final**3 - t_init**3)/3.

class SineSquared:
    __name__ = "SineSquared"
    def __init__(self, freqency) -> None:
        self.w = freqency
        self.w_sol = 2*self.w
    
    def __call__(self, t, y=None):
        return torch.sin(t*self.w)**2

    def solve(self, t_init, t_final):
        solution = torch.sin(torch.tensor([self.w_sol*t_init]))
        solution -= torch.sin(torch.tensor([self.w_sol*t_final]))
        return (t_final - t_init)/2. + solution/(2*self.w_sol)

class Exponential:
    __name__ = "Exponential"
    def __init__(self, decay_constant) -> None:
        self.t_const = decay_constant
    
    def __call__(self, t, y=None):
        return torch.exp(self.t_const*t)

    def solve(self, t_init, t_final):
        return (torch.exp(torch.tensor([t_final*self.t_const]))\
            - torch.exp(torch.tensor([t_init*self.t_const])))/self.t_const

class DampedSine:
    __name__ = "DampedSine"
    def __init__(self, freqency, decay_constant):
        assert decay_constant > 0
        self.w = freqency
        self.t_const = decay_constant

    def __call__(self, t, y=None):
        return torch.exp(-self.t_const*t)*torch.sin(self.w*t)

    def _numerator(self, t):
        t = torch.tensor([t])
        return -1*torch.exp(-self.t_const*t)*(self.t_const*torch.sin(self.w*t) + self.w*torch.cos(self.w*t))
    
    def solve(self, t_init, t_final):
        denomenator = self.t_const**2 + self.w**2
        return (self._numerator(t_final) - self._numerator(t_init))/denomenator