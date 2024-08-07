import torch
from einops import rearrange

def _add_initial_y(ode_fxn, t, t_step_fxn, t_init=0., t_final=1.):
    #t_steps = t_spacing_fxn(t[:-1], t[1:])
    t_steps = t_step_fxn(t)
    print("INIT T", t_steps.shape)
    n_intervals = int(t_steps.shape[1])

    t_add = torch.concatenate(
        [t_steps[0], t_steps[1:,1:].reshape((-1, *(t_steps.shape[2:])))],
        dim=0
    )
    
    # Calculate new geometries
    print("T ADD", t_add.shape)
    y_add = ode_fxn(t_add)
    print("Y add", y_add.shape)

    y_steps = torch.reshape(y_add[1:], (len(y_add)-1, n_intervals-1, -1))
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


def _adaptively_add_y(
        ode_fxn, y, t, error_ratios, t_step_fxn, t_y_fusion, t_init, t_final
    ):
    """
    t : [t-1, p1]
    y : [t-1, p1]
    t_add : [ta] time between t[idx-1] and t[idx]
    idxs_add : [ta] index where it should be given current t
    """


    if y is None:
        return _add_initial_y(ode_fxn=ode_fxn, t=t, t_step_fxn=t_step_fxn, t_init=t_init, t_final=t_final)

    print("ADAPTIVE ADD INP SHAPES", y.shape, t.shape)
    if len(torch.where(error_ratios > 1.)[0]) == 0:
        return t, y, error_ratios
         
    # Get new time steps for merged steps
    """
    if idxs_remove_pair is not None:
        t_steps_remove = t_step_fxn(t, idx_remove_pair)
        #    t[idxs_remove_pair,0], t[idxs_remove_pair+1,-1] 
        #)
        t_steps_remove = t_steps_remove[:,1:-1]
    
    """
    idxs_add = torch.where(error_ratios > 1.)[0]
    print("idxs add", t.shape, idxs_add.shape, idxs_add)
    t_steps_add = (t[idxs_add,1:] +  t[idxs_add,:-1])/2     #[n_add, p, 1]
    print("t add", t_steps_add.shape)
    #t_steps_add = t_step_fxn(t, idxs_add)
    ##t_steps_add = t_steps_add[:,:-1]

    """
    t_bisect = (t[idxs_add,0] + t[idxs_add,-1])/2.
    t_left = torch.concatenate(
        [t[idxs_add,0].unsqueeze(1), t_bisect.unsqueeze(1)], dim=1
    )
    t_right = torch.concatenate(
        [t_bisect.unsqueeze(1), t[idxs_add,-1].unsqueeze(1)], dim=1
    )
    t_steps_add = t_spacing_fxn(t_left.flatten(), t_right.flatten())
    t_steps_add = t_steps_add[:,:-1]
    """ 

    # Calculate new geometries
    n_add, p, d = t_steps_add.shape
    # ode_fxn input is 2 dims, t_steps_add has 3 dims, combine first two
    t_steps_add = rearrange(t_steps_add, 'n p d -> (n p) d')
    y_add = ode_fxn(t_steps_add)
    t_steps_add = rearrange(t_steps_add, '(n p) d -> n p d', n=n_add, p=p) 
    y_add = rearrange(y_add, '(n p) d -> n p d', n=n_add, p=p) 

    """
    t_add, y_add = t_y_fusion(idxs_add, t, t_add, y, y_add)
    y_add = rearrange(y_add, '(a b) p -> a b p', b=2)
    t_steps_add = rearrange(t_steps_add, '(a b) p -> a b p', b=2)
    """

    """
    y_steps = torch.reshape(y_add[1:], (-1, n_intervals-1))
    y_steps = torch.concatenate(
        [
            torch.concatenate(
                [torch.tensor([y_add[0]]), y_steps[:-1,-1]]
            ).unsqueeze(1),
            y_steps
        ],
        dim=1
    )
    """

    # Create new vector to fill with old and new values
    y_combined = torch.zeros(
        (len(y)+len(y_add), p+1, y_add.shape[-1]),
        requires_grad=False
    ).detach()
    t_combined = torch.zeros_like(y_combined, requires_grad=False).detach() - 1
    
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
    t_add_combined = torch.ones((len(idxs_add), (p+1)*2-1, d))*torch.nan
    t_add_combined[:,torch.arange(p+1)*2] = t[idxs_add]
    t_add_combined[:,torch.arange(p)*2+1] = t_steps_add
    t_combined[idxs_add_offset,:,:] = t_add_combined[:,:p+1]
    t_combined[idxs_add_offset+1,:,:] = t_add_combined[:,p:]

    print("TESTING T COMBINED", t[:,:,0], '\n',t_combined[:,:,0])
    y_add_combined = torch.ones((len(idxs_add), (p+1)*2-1, d))*torch.nan
    y_add_combined[:,torch.arange(p+1)*2] = y[idxs_add]
    y_add_combined[:,torch.arange(p)*2+1] = y_add
    y_combined[idxs_add_offset,:,:] = y_add_combined[:,:p+1]
    y_combined[idxs_add_offset+1,:,:] = y_add_combined[:,p:]

    #y_combined[idx_add,:-1] = y_add[:,0]
    #y_combined[idx_add+1,:-1] = y_add[:,1]
    #t_combined[idx_add,:-1] = t_steps_add[:,0]
    #t_combined[idx_add+1,:-1] = t_steps_add[:,1]

    #y_combined[:-1,-1] = y_combined[1:,0]
    #t_combined[:-1,-1] = t_combined[1:,0]

    return y_combined, t_combined


    if len(t_add) == 0:
        RuntimeWarning("Do not expect empty points to add.")
        return y, t
    if y is None:
        y = torch.tensor([])
    if t is None:
        y = torch.tensor([])
    
    # Calculate new geometries
    y_add = ode_fxn(t_add)
    
    # Place new geometries between existing 
    y_combined = torch.zeros(
        (len(y)+len(y_add), y_add.shape[-1]),
        requires_grad=False
    ).detach()
    y_idxs = None
    if y is not None and len(y):
        y_idxs = torch.arange(len(y), dtype=torch.int)
        bisected_idxs = idxs_add - torch.arange(len(idxs_add))
        for i, idx in enumerate(bisected_idxs[:-1]):
            y_idxs[idx:bisected_idxs[i+1]] += i + 1
        y_idxs[bisected_idxs[-1]:] += len(bisected_idxs)
        y_combined[y_idxs] = y
    assert(torch.all(y_combined[idxs_add] == 0))
    y_combined[idxs_add] = y_add

    # Place new times between existing 
    t_combined = torch.zeros(
        (len(y)+len(t_add), 1), requires_grad=False
    )
    if y_idxs is not None:
        t_combined[y_idxs] = t
    t_combined[idxs_add] = t_add

    return y_combined, t_combined


def __adaptively_add_y(ode_fxn, y, t, error_ratios=None, t_add=None, idxs_add=None):
    """
    t : [t]
    y : [t-1, p1]
    t_add : [ta] time between t[idx-1] and t[idx]
    idxs_add : [ta] index where it should be given current t
    """

    # Check inputs, either error_ratios, or t_add and idx_add

    if error_ratios is not None:
        raise NotImplementedError
    #else:

    if len(t_add) == 0:
        RuntimeWarning("Do not expect empty points to add.")
        return y, t
    if y is None:
        y = torch.tensor([])
    if t is None:
        y = torch.tensor([])
    
    # Calculate new geometries
    y_add = ode_fxn(t_add)
    
    # Place new geometries between existing 
    y_combined = torch.zeros(
        (len(y)+len(y_add), y_add.shape[-1]),
        requires_grad=False
    ).detach()
    y_idxs = None
    if y is not None and len(y):
        y_idxs = torch.arange(len(y), dtype=torch.int)
        bisected_idxs = idxs_add - torch.arange(len(idxs_add))
        for i, idx in enumerate(bisected_idxs[:-1]):
            y_idxs[idx:bisected_idxs[i+1]] += i + 1
        y_idxs[bisected_idxs[-1]:] += len(bisected_idxs)
        y_combined[y_idxs] = y
    assert(torch.all(y_combined[idxs_add] == 0))
    y_combined[idxs_add] = y_add

    # Place new times between existing 
    t_combined = torch.zeros(
        (len(y)+len(t_add), 1), requires_grad=False
    )
    if y_idxs is not None:
        t_combined[y_idxs] = t
    t_combined[idxs_add] = t_add

    return y_combined, t_combined


def _remove_excess_y(t, error_ratios_2steps, remove_cut, remove_fxn):
        
    #ratio_idxs_cut = torch.where(error_ratios < remove_cut)[0]
    ratio_mask_cut = error_ratios_2steps < remove_cut
    if len(error_ratios_2steps) == 0:
        return torch.empty(0, dtype=torch.long)
    # Since error ratios encompasses 2 RK steps each neighboring element shares
    # a step, we cannot remove that same step twice and therefore remove the 
    # first in pair of steps that it appears in
    #print("RC1", ratio_mask_cut)
    ratio_mask_cut = torch.concatenate(
        [_rec_remove(error_ratios_2steps < remove_cut), torch.tensor([False])]
    ).to(torch.bool)
    print("RATIO MASK CuT", ratio_mask_cut.shape, t.shape)
    #ratio_idxs_cut = torch.where(ratio_mask_cut)[0] # Index for first interval of 2
    print(ratio_mask_cut, ratio_mask_cut[:-1]*ratio_mask_cut[1:])
    assert not torch.any(ratio_mask_cut[:-1]*ratio_mask_cut[1:])

    if not any(ratio_mask_cut):
        return t
    
    #ratio_idxs_cut = torch.concatenate(
    #    [ratio_idxs_cut.unsqueeze(1), 1+ratio_idxs_cut.unsqueeze(1)], dim=1
    #)
    t_pruned = remove_fxn(t, ratio_mask_cut)

    return t_pruned

def __find_excess_y(p, error_ratios, remove_cut):
        
    #ratio_idxs_cut = torch.where(error_ratios < remove_cut)[0]
    ratio_mask_cut = error_ratios < remove_cut
    if len(error_ratios) <= 1:
        return torch.zeros([len(error_ratios)], dtype=bool)
    # Since error ratios encompasses 2 RK steps each neighboring element shares
    # a step, we cannot remove that same step twice and therefore remove the 
    # first in pair of steps that it appears in
    #print("RC1", ratio_mask_cut)
    ratio_mask_cut = _rec_remove(error_ratios < remove_cut)
    ratio_idxs_cut = torch.where(ratio_mask_cut)[0]

    # Remove every other intermediate evaluation point
    ratio_idxs_cut = p*ratio_idxs_cut + 1
    ratio_idxs_cut = torch.flatten(
        ratio_idxs_cut.unsqueeze(1) + 2*torch.arange(p).unsqueeze(0)
    )
    y_mask_cut = torch.zeros((len(error_ratios)+1)*p+1, dtype=torch.bool)
    y_mask_cut[ratio_idxs_cut] = True

    return y_mask_cut
        
"""
    deltas = self._geo_deltas(geos)
    remove_mask = deltas < self.dxdx_remove
    #print("REMOVE DELTAS", deltas[:10])
    while torch.any(remove_mask):
        # Remove largest time point when geo_delta < dxdx_remove
        remove_mask = torch.concatenate(
            [
                torch.tensor([False]), # Always keep t_init
                remove_mask[:-2],
                torch.tensor([remove_mask[-1] or remove_mask[-2]]),
                torch.tensor([False]), # Always keep t_final
            ]
        )
        #print("N REMoVES", torch.sum(remove_mask), remove_mask[:10])

        #print("test not", remove_mask, ~remove_mask)
        eval_times = eval_times[~remove_mask]
        geos = geos[~remove_mask]
        deltas = self._geo_deltas(geos)
        remove_mask = deltas < self.dxdx_remove
    
    if len(eval_times) == 2:
        print("WARNING: dxdx is too large, all integration points have been removed")
    
    return geos, eval_times
"""

def _rec_remove(mask):
    mask2 = mask[:-1]*mask[1:]
    if torch.any(mask2):
        if mask2[0]:
            mask[1] = False
        if len(mask) > 2:
            return _rec_remove(torch.concatenate(
                [
                    mask[:2],
                    mask2[1:]*mask[:-2] + (~mask2[1:])*mask[2:]
                ]
            ))
        else:
            return mask
    else:
        return mask



def __remove_idxs_to_ranges(idxs_cut):
    ranges_cut = []
    range_i = idxs_cut[0]
    idxC = 0
    while idxC < len(idxs_cut):
        for idxP in range(5):
            if idxs_cut[idxC+1] - idxs_cut[idxC] > 1:
                ranges_cut.append((range_i, idxs_cut[idxC]))
                range_i = idxs_cut[idxC+1]
        idxC += 1
    ranges_cut.append((range_i, idxs_cut[-1]))

    return ranges_cut

   
def _remove_idxs_to_ranges(idxs_cut):
    ranges_cut = []
    range_i = idxs_cut[0]
    idxC = 0
    while idxC < len(idxs_cut) - 1:
        if idxs_cut[idxC+1] - idxs_cut[idxC] > 1:
            ranges_cut.append((range_i, idxs_cut[idxC]))
            range_i = idxs_cut[idxC+1]
        idxC += 1
    ranges_cut.append((range_i, idxs_cut[-1]))

    return ranges_cut

def _find_sparse_y(t, p, error_ratios):
    #print("SPARSE ERROR RATIO", error_ratios)
    ratio_idxs_cut = torch.where(error_ratios > 1.)[0]
    #print("ratio idxs", ratio_idxs_cut[-10:])
    #ratio_idxs_cut = p*ratio_idxs_cut + 1
    ratio_idxs_cut = p*ratio_idxs_cut + 1
    idxs_add = torch.flatten(
        ratio_idxs_cut.unsqueeze(1) + torch.arange(p).unsqueeze(0)
        #ratio_idxs_cut.unsqueeze(1) + 2*torch.arange(p).unsqueeze(0)
    )

    #print(ratio_idxs_cut.unsqueeze(1) + 2*torch.arange(p).unsqueeze(0))
    #print(len(error_ratios), len(t), idxs_add[-10:])
    t_add = (t[idxs_add-1] + t[idxs_add])/2
    idxs_add += torch.arange(len(idxs_add)) # Account for previosly added points
    
    return t_add, idxs_add


def _compute_error_ratios(sum_step_p, sum_step_p1, rtol, atol, norm):
    error_estimate = torch.abs(sum_step_p1 - sum_step_p)
    error_tol = atol + rtol*torch.max(sum_step_p.abs(), sum_step_p1.abs())
    error_ratio = norm(error_estimate/error_tol).abs()
    print("COMP1", error_ratio.shape)

    error_estimate_2steps = error_estimate[:-1] + error_estimate[1:]
    error_tol_2steps = atol + rtol*torch.max(
        torch.stack(
            [sum_step_p[:-1].abs(), sum_step_p[1:].abs(), sum_step_p1[:-1].abs(), sum_step_p1[1:].abs()]
        ),
        dim=0
    )[0]
    error_ratio_2steps= norm(error_estimate_2steps/error_tol_2steps).abs() 
    
    return error_ratio, error_ratio_2steps