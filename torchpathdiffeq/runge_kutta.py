import torch
from .base import steps, get_sampling_type, MethodOutput
from .parallel_solver import ParallelVariableAdaptiveStepsizeSolver, ParallelUniformAdaptiveStepsizeSolver



def _RK_integral(
        t, 
        y, 
        tableau_b, 
        y0=torch.tensor([0], dtype=torch.float64),
        verbose=False
    ):
    """
    Performs a single Runge-Kutta sum over the p or more evaluations of
    ode_fxn, where p is the order of the Runge-Kutta integration method.

    Args:
        t (Tensor): Current time evaluations in the path integral
        y (Tensor): Evaluations of ode_fxn at time points t
        tableau_b (Tensor): Tableau b values (see wiki for notation convention)
            that weight the temporal components after summing over y steps
        y0 (Tensor): The initial integral value
        verbose (bool): Boolean for printing intermediate results
    
    Shapes:
        t: [N, C, T]
        y: [N, C, D]
        tableau_b: [N, C, 1] or [1, C, 1]
        y0: [D]
    """
    h = t[:,-1] - t[:,0]
    if verbose:
        print("H", h.shape, h)
    
    RK_steps = h*torch.sum(tableau_b*y, dim=1)   # Sum over k evaluations weighted by c
    if verbose:
        print("RK STEPS", RK_steps.shape, RK_steps)
    integral = y0 + torch.sum(RK_steps)                    # Sum over all steps with step size h
    return integral, RK_steps, h

   
class RKParallelUniformAdaptiveStepsizeSolver(ParallelUniformAdaptiveStepsizeSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calculate_integral(self, t, y, y0=0):
        """
        Internal Runge-Kutta (RK) integration method, carries out the RK method
        on the given time (t) and ode_fxn evaluation points (y).

        Args:
            t (Tensor): Evaluation time steps for the RK integral
            y (Tensor): Evalutions of the integrad at time steps t
            y0 (Tensor): Initial values of the integral
        
        Shapes:
            t: [N, C, T]
            y: [N, C, D]
            y0: [D]
        """
        tableau_b, tableau_b_error = self._get_tableau_b(t)
        integral, RK_steps, h = _RK_integral(t, y, tableau_b, y0=y0)
        integral_error, step_errors, _ = _RK_integral(t, y, tableau_b_error, y0=y0)
        return MethodOutput(
            integral=integral,
            integral_error=integral_error,
            sum_steps=RK_steps,
            sum_step_errors=step_errors,
            h=h
        )
    

    def _get_tableau_b(self, t):
        """
        Return the Tableau b values (see wiki for notation convention)that
        weight the temporal components after summing over y steps

        Args:
            t (Tensor): Current time evaluations in the path integral
        
        Shapes:
            t: [N, C, T]
        """
        return self.method.tableau.b.unsqueeze(-1),\
            self.method.tableau.b_error.unsqueeze(-1)
    

    def _get_num_tableau_c(self):
        """
        Return the number of Tableau c values (see wiki for notation
        convention) that determine the fractional time evaluation 
        points within a single RK step.
        """
        return len(self.method.tableau.c)


class RKParallelVariableAdaptiveStepsizeSolver(ParallelVariableAdaptiveStepsizeSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calculate_integral(self, t, y, y0=0):
        """
        Internal Runge-Kutta (RK) integration method, carries out the RK method
        on the given time (t) and ode_fxn evaluation points (y).

        Args:
            t (Tensor): Evaluation time steps for the RK integral
            y (Tensor): Evalutions of the integrad at time steps t
            y0 (Tensor): Initial values of the integral
        
        Shapes:
            t: [N, C, T]
            y: [N, C, D]
            y0: [D]
        """
        tableau_b, tableau_b_error = self._get_tableau_b(t)
        integral, RK_steps, h = _RK_integral(t, y, tableau_b, y0=y0)
        integral_error, step_errors, _ = _RK_integral(t, y, tableau_b_error, y0=y0)
        return MethodOutput(
            integral=integral,
            integral_error=integral_error,
            sum_steps=RK_steps,
            sum_step_errors=step_errors,
            h=h
        )
    
    def _get_tableau_b(self, t):
        """
        Return the Tableau b values (see wiki for notation convention)that
        weight the temporal components after summing over y steps

        Args:
            t (Tensor): Current time evaluations in the path integral
        
        Shapes:
            t: [N, C, T]
        """
        norm_dt = t - t[:,0,None]
        norm_dt = norm_dt/norm_dt[:,-1,None]
        b, b_error = self.method.tableau_b(norm_dt)
        return b.unsqueeze(-1), b_error.unsqueeze(-1)
    
    def _get_num_tableau_c(self):
        """
        Return the number of Tableau c values (see wiki for notation
        convention) that determine the fractional time evaluation 
        points within a single RK step.
        """
        return self.method.n_tableau_c


def get_parallel_RK_solver(sampling_type, *args, **kwargs):
    """
    Return either the uniform or variable sampling RK method given input args.
    """
    if isinstance(sampling_type, str):
        sampling_type = get_sampling_type(sampling_type)
    if sampling_type == steps.ADAPTIVE_UNIFORM:
        return RKParallelUniformAdaptiveStepsizeSolver(*args, **kwargs)
    elif sampling_type == steps.ADAPTIVE_VARIABLE:
        return RKParallelVariableAdaptiveStepsizeSolver(*args, **kwargs)
    else:
        raise ValueError()