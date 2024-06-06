import numpy as np

from src.tools import visualize


class logging():
    def __init__(self, *args, **kwargs):
        return
    
    def training_logger(self, step, val):
        step_string = ("step: " + str(step)).ljust(15)
        val_string = "val: " + str(val)
        print(step_string, val_string)
    
    def optimization_step(
            self,
            step,
            path,
            potential,
            path_integral,
            plot=True,
            geo_paths=None,
            pes_paths=None,
            add_azimuthal_dof=False,
            add_translation_dof=False
        ):
        print(f"Step {step} | Loss: {path_integral.integral.item():.7}")
        """
        for ii in range(len(path.mlp.layers)):
            print(f"W{ii} sum: {jnp.sum(path.mlp.layers[ii].weight)}")
            print(f"Wg{ii} sum: {jnp.sum(grads.mlp.layers[ii].weight)}")
        """
        #print(path.mlp.layers[0].weight)
        #print("test grad", grads.mlp.layers[0].weight)
        path_output = path.get_path(return_velocity=True, return_force=True)
        #print('PATH SHAPE', geo_path.shape, pes_path.shape)
        if geo_paths is not None:
            geo_paths.append(path_output.geometric_path)
        if pes_paths is not None:
            pes_paths.append(path_output.potential_path)
        if plot:
            visualize.plot_path(
                path_output.geometric_path.detach().to('cpu').numpy(),
                f"test_plot_{step:03d}",
                pes_fxn=potential,
                plot_min_max=(-2, 2, -2, 2),
                levels=np.arange(-100, 100, 5),
                time_history=path_integral.times,
                geometry_history=path_integral.geometry,
                add_translation_dof=add_translation_dof,
                add_azimuthal_dof=add_azimuthal_dof
            )
        return path_output