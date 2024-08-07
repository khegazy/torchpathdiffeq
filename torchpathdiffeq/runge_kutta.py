import torch
import numpy as np
from numpy.testing import assert_allclose
from . import tableau
from .solvers import degree, steps, ParallelAdaptiveStepsizeSolver

TABLEAU_CALCULATORS = {
    'euler' : (tableau.euler_tableau_b, tableau.heun_tableau_b),
    'heun' : (tableau.euler_tableau_b, tableau.heun_tableau_b),
    'generic3' : (tableau.heun_tableau_b, tableau.generic3_tableau_b)
}

class RKParallelAdaptiveStepsizeSolver(ParallelAdaptiveStepsizeSolver):
    def __init__(self, solver, atol, rtol, y0=torch.tensor([0], dtype=torch.float), remove_cut=0.1, ode_fxn=None, t_init=0., t_final=1.):
        super().__init__(
            solver=solver, atol=atol, rtol=rtol, remove_cut=remove_cut, ode_fxn=ode_fxn, t_init=t_init, t_final=t_final
        )
        class_tb_p, class_tb_p1 = TABLEAU_CALCULATORS[solver]
        self.tableau_b_p = class_tb_p()
        self.tableau_b_p1 = class_tb_p1()
        self.y0 = y0
    

    def _calculate_integral(self, t, y, y0=0, degr=degree.P1):
        verbose = False
        """
        dt = t[1:] - t[:-1]
        _t_p = t[0::self.p]
        h = _t_p[1:] - _t_p[:-1]
        """
        h = t[:,-1] - t[:,0]
        if verbose:
            print("H", h.shape, h)
        
        tableau_b = self._calculate_tableau_b(t, degr=degr)
        print("IN CALC INT H / TB", h.shape, tableau_b.shape)
        assert(tableau_b.shape[1] == max(2, self.p+1))
        #assert(torch.all(torch.sum(tableau_b, dim=-1) == 1))
        
        # The last point in the h-1 RK step is the first point in the h RK step
        """
        combined_tableau_b = tableau_b
        combined_tableau_b[:-1,-1] += tableau_b[1:,0]
        combined_tableau_b = torch.concatenate(
            [combined_tableau_b[0], torch.flatten(combined_tableau_b[1:,1:])]
        )
        """
        
        """
        #y_steps = torch.reshape(y[1:], (-1, max(self.p-1, 1)))
        y_steps = torch.reshape(y[1:], (-1, max(self.p, 1)))
        y_steps = torch.concatenate(
            [
                torch.concatenate(
                    [torch.tensor([y[0]]), y_steps[:-1,-1]]
                ).unsqueeze(1),
                y_steps
            ],
            dim=1
        )
        """
        y_steps = y

        #print("SHAPES dt / h / tb / ysteps", dt.shape, h.shape, tableau_b.shape, y_steps.shape)
        # Shapes
        # dt: [t-1]
        # h: [dt/p x 1]
        # tableau_b: [dt/p x p+1] check
        # y_steps: [dt/p x p+1] check
        print("TABLEAU / YSTEPS / H", tableau_b.shape, y_steps.shape, h.shape)
        RK_steps = h*torch.sum(tableau_b*y_steps, dim=1)   # Sum over k evaluations weighted by c
        print("RK1", RK_steps.shape)
        if verbose:
            print("RK STEPS", RK_steps.shape, RK_steps)
        integral = y0 + torch.sum(RK_steps)                    # Sum over all steps with step size h
        print("INT / RK STEPS / H", integral.shape, RK_steps.shape, h.shape)
        return integral, RK_steps, h


    def _calculate_tableau_b(self, t, degr):
        if degr == degree.P:
            tableau_b_fxn = self.tableau_b_p
        elif degr == degree.P1:
            tableau_b_fxn = self.tableau_b_p1
        
        norm_dt = t - t[:,0,None]
        norm_dt = norm_dt/norm_dt[:,-1,None]
        b = tableau_b_fxn(norm_dt, degr).unsqueeze(-1)
        assert np.all(np.abs(torch.sum(b, dim=1).numpy() - 1.) < 1e-5)
        #assert_allclose(torch.sum(b, dim=-1).numpy(), 1.)

        return b
