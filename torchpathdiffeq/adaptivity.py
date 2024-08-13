import torch
from einops import rearrange

class IntegralAdaptivity():
    def __init__(self):
        self.remove_cut = None
        self.rtol = None
        self.atol = None

    def _initial_t_steps(
            self,
            t,
            t_init=torch.tensor([0], dtype=torch.float64),
            t_final=torch.tensor([1], dtype=torch.float64)
        ):
        """
        Creates an initial time sampling tensor either from scratch or from a
        tensor of time points with dimension d.

        Args:
            t (Tensor): Input time, either None or tensor starting and
                ending at the integration bounds
            t_init (Tensor, optional): Minimum of integral range
            t_final (Tensor, optional): Maximum of integral range
        
        Shapes:
            t : [N, T] will populate intermediate evaluations according to
                integration method, [N, C, T] will retun t
            t_init: [T]
            t_final: [T]
        """
        raise NotImplementedError
    

    def _remove_excess_t(self, t, remove_idxs):
        """
        Merges neighboring time steps or removes and one time steps and extends
        its neighbor to cover the same range.

        Args:
            t (Tensor): Integration time steps
            remove_idxs (Tensor): First index of neighboring steps needed to be
                merged, or remove at given index and extend the following step
        
        Shapes:
            t : [N, C, T]
            removed_idxs : [n]
        """
        raise NotImplementedError
    

    def _error_norm(self, error):
        """
        Normalize multivariate errors to determine the step's total error
        """
        raise NotImplementedError
    

    def _add_initial_y(
            self,
            ode_fxn,
            t,
            t_init=torch.tensor([0], dtype=torch.float64),
            t_final=torch.tensor([1], dtype=torch.float64),
            ode_args=()
        ):
        """
        Initial evaluation of ode_fxn for the given t points, if t is None it
        will be initialized within the range given by t_init and t_final

        Args:
            ode_fxn (Callable): The ode function to be integrated
            t (Tensor): Initial time evaluations in the path integral, either
                None or tensor starting and ending at the integration bounds
            t_init (Tensor, optional): Minimum of integral range
            t_final (Tensor, optional): Maximum of integral range
            ode_args (Tuple): Arguments for ode_fxn
        
        Shapes:
            t : [N, C, T]
            t_init: [T]
            t_final: [T]
        
        Notes:
            ode_fxn takes as input (t, *args)
        """
        if t is not None and len(t.shape) == 1:
            t = t.unsqueeze(-1)
        #t_steps = t_spacing_fxn(t[:-1], t[1:])
        t_steps = self._initial_t_steps(
            t, t_init=t_init, t_final=t_final
        ).to(torch.float64)
        print("INIT T", t_steps.shape)
        n, n_c, d = t_steps.shape

        print("INIT T", t_steps[:,:,0])
        t_add = torch.concatenate(
            [t_steps[0], t_steps[1:,1:].reshape((-1, *(t_steps.shape[2:])))],
            dim=0
        )
        
        # Calculate new geometries
        print("T ADD", t_add.shape)
        y_add = ode_fxn(t_add, *ode_args).to(torch.float64)
        print("Y add", y_add.shape)

        y_steps = torch.reshape(y_add[1:], (n, n_c-1, -1))
        y_steps = torch.concatenate(
            [
                torch.concatenate(
                    [y_add[None,None,0,...], y_steps[:-1,None,-1]], dim=0
                ),
                y_steps
            ],
            dim=1
        )

        print("OUT T", t_steps[:,:,0])
        print("OUT T Y", t_steps.shape, y_steps.shape)
        return y_steps, t_steps


    def adaptively_add_y(
            self,
            ode_fxn,
            y=None,
            t=None,
            error_ratios=None,
            t_init=torch.tensor([0], dtype=torch.float64),
            t_final=torch.tensor([1], dtype=torch.float64),
            ode_args=()
        ):
        """
        Adds new time points between current time points and splits these
        poinsts into two steps where error_ratio < 1. ode_fxn is evaluated at
        these new time points, both the new time points and y points are merged
        with the original values in order of time.

        Args:
            ode_fxn (Callable): The ode function to be integrated
            y (Tensor): Evaluations of ode_fxn at time points t
            t (Tensor): Current time evaluations in the path integral
            error_ratios (Tensor): Numerical integration error ratios for each
                integration step
            t_init (Tensor, optional): Minimum of integral range
            t_final (Tensor, optional): Maximum of integral range
            ode_args (Tuple): Arguments for ode_fxn
        
        SHAPES:
            y: [N, C, D]
            t : [N, C, T]
            error_ratios : [N]
            t_init: [T]
            t_final: [T]
        
        Notes:
            ode_fxn takes as input (t, *args)
        """
        if y is None or t is None or error_ratios is None:
            return self._add_initial_y(ode_fxn=ode_fxn, ode_args=ode_args, t=t, t_init=t_init, t_final=t_final)

        print("ADAPTIVE ADD INP SHAPES", y.shape, t.shape)
        if torch.all(error_ratios <= 1.):
            return y, t
            
        # Get new time steps for merged steps
        idxs_add = torch.where(error_ratios > 1.)[0]
        print("idxs add", t.shape, idxs_add.shape, idxs_add)
        t_steps_add = (t[idxs_add,1:] +  t[idxs_add,:-1])/2     #[n_add, p, 1]
        print("T and add in adpative", t.shape, t_steps_add.shape, '\n', t[:,:,0], '\n', t_steps_add[:,:,0])
        #t_steps_add = t_step_fxn(t, idxs_add)
        ##t_steps_add = t_steps_add[:,:-1]

        # Calculate new geometries
        n_add, nm1_c, d = t_steps_add.shape
        # ode_fxn input is 2 dims, t_steps_add has 3 dims, combine first two
        t_steps_add = rearrange(t_steps_add, 'n p d -> (n p) d')
        y_add = ode_fxn(t_steps_add, *ode_args).to(torch.float64)
        t_steps_add = rearrange(t_steps_add, '(n c) d -> n c d', n=n_add, c=nm1_c) 
        y_add = rearrange(y_add, '(n c) d -> n c d', n=n_add, c=nm1_c) 

        # Create new vector to fill with old and new values
        y_combined = torch.zeros(
            (len(y)+len(y_add), nm1_c+1, y_add.shape[-1]),
            dtype=torch.float64,
            requires_grad=False
        ).detach()
        t_combined = torch.zeros_like(
            y_combined, dtype=torch.float64, requires_grad=False
        ).detach()
        
        # Add old t and y values, skipping regions with new points
        idx_offset = torch.zeros(len(y), dtype=torch.long)
        idx_offset[idxs_add] = 1
        idx_offset = torch.cumsum(idx_offset, dim=0)
        idx_input = torch.arange(len(y)) + idx_offset
        y_combined[idx_input,:] = y
        t_combined[idx_input,:] = t

        # Add new t and y values to added rows
        print("BEGINING COMB", y.shape, t.shape, y_combined.shape)
        idxs_add_offset = idxs_add + torch.arange(len(idxs_add))
        t_add_combined = torch.nan*torch.ones(
            (len(idxs_add), (nm1_c+1)*2-1, d), dtype=torch.float64
        )
        t_add_combined[:,torch.arange(nm1_c+1)*2] = t[idxs_add]
        t_add_combined[:,torch.arange(nm1_c)*2+1] = t_steps_add
        t_combined[idxs_add_offset,:,:] = t_add_combined[:,:nm1_c+1]
        t_combined[idxs_add_offset+1,:,:] = t_add_combined[:,nm1_c:]

        print("TESTING T COMBINED", t[:,:,0], '\n',t_combined[:,:,0])
        y_add_combined = torch.nan*torch.ones(
            (len(idxs_add), (nm1_c+1)*2-1, d), dtype=torch.float64
        )
        y_add_combined[:,torch.arange(nm1_c+1)*2] = y[idxs_add]
        y_add_combined[:,torch.arange(nm1_c)*2+1] = y_add
        y_combined[idxs_add_offset,:,:] = y_add_combined[:,:nm1_c+1]
        y_combined[idxs_add_offset+1,:,:] = y_add_combined[:,nm1_c:]

        assert torch.all(t_combined[:-1,-1] == t_combined[1:,0])
        return y_combined, t_combined


    def remove_excess_y(self, t, error_ratios_2steps):
        """
        Remove a single integration time step where
        error_ratios_2steps < remove_cut by merging two neighboring time steps,
        error_ratios_2steps corresponds to the first time step of the pair.
        This function only alters t, where remove_fxn merges the two steps.

        Args:
            t (Tensor): Current time evaluations in the path integral
            error_ratios_2steps (Tensor): The merged errors of neighboring time
                steps, these indices align with the first step of the pair
                (error_ratios_2steps[i] -> t[i])
        
        Shapes:
            t: [N, C, T]
            error_ratios_2steps: [N-1]
        """
            
        #ratio_idxs_cut = torch.where(error_ratios < remove_cut)[0]
        ratio_mask_cut = error_ratios_2steps < self.remove_cut
        if len(error_ratios_2steps) == 0:
            return torch.empty(0, dtype=torch.long)
        # Since error ratios encompasses 2 RK steps each neighboring element shares
        # a step, we cannot remove that same step twice and therefore remove the 
        # first in pair of steps that it appears in
        #print("RATIO MASK CuT", ratio_mask_cut.shape, t.shape)
        ratio_idxs_cut = torch.where(self._rec_remove(error_ratios_2steps < self.remove_cut))[0] # Index for first interval of 2
        #print(ratio_mask_cut, ratio_mask_cut[:-1]*ratio_mask_cut[1:])
        print("RATIO idxs cut", ratio_idxs_cut)
        assert not torch.any(ratio_idxs_cut[:-1] + 1 == ratio_idxs_cut[1:])

        if len(ratio_idxs_cut) == 0:
            return t
        
        #ratio_idxs_cut = torch.concatenate(
        #    [ratio_idxs_cut.unsqueeze(1), 1+ratio_idxs_cut.unsqueeze(1)], dim=1
        #)
        t_pruned = self._remove_excess_t(t, ratio_idxs_cut)

        return t_pruned

    def _rec_remove(self, mask):
        """
        Make no neighboring values are both True by setting the second value
        to False, this is done recursively.
        """
        mask2 = mask[:-1]*mask[1:]
        if torch.any(mask2):
            if mask2[0]:
                mask[1] = False
            if len(mask) > 2:
                return self._rec_remove(torch.concatenate(
                    [
                        mask[:2],
                        mask2[1:]*mask[:-2] + (~mask2[1:])*mask[2:]
                    ]
                ))
            else:
                return mask
        else:
            return mask


    def compute_error_ratios(self, sum_steps, sum_step_errors):
        """
        Computes the ratio of the difference between chosen method of order p
        and a method of order p-1, and the error tolerance determined by atol
        and rtol. Integration steps of order p-1 use the same points.

        Args:
            sum_steps (Tensor): Sum over all t and y evaluations in a single
                RK step multiplied by the total delta t for that step (h)
            sum_step_errors (Tensor): Similar to sum_steps but evaluated with
                and error tableau made of the differences between a method of
                order p and one of order p-1
        
        Shapes:
            sum_steps: [N, T]
            sum_step_errors: [N, T]

        """
        print("TODO: Removed max in error_tol, check against torchdiffeq")
        error_estimate = torch.abs(sum_step_errors)
        error_tol = self.atol + self.rtol*torch.abs(sum_steps)
        error_ratio = self._error_norm(error_estimate/error_tol).abs()
        print("COMP1", error_ratio.shape)

        error_estimate_2steps = error_estimate[:-1] + error_estimate[1:]
        error_tol_2steps = self.atol + self.rtol*torch.max(
            torch.stack(
                [sum_steps[:-1].abs(), sum_steps[1:].abs()]
            ),
            dim=0
        )[0]
        error_ratio_2steps= self._error_norm(error_estimate_2steps/error_tol_2steps).abs() 
        
        return error_ratio, error_ratio_2steps