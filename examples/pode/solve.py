import torch
import numpy as np
from torch import nn
import torchpathdiffeq as tpd


class BaseODE():
    def __init__(self, N_dims):
        self.N_dims = N_dims
    
    def __call__(self, t):
        raise NotImplementedError

class linear(BaseODE):
    def __init__(self):
        super().__init__(1)

    def __call__(self, t):
        return t

class quadratic(BaseODE):
    def __init__(self):
        super().__init__(1)
    
    def __call__(self, t):
        return t**2


class CurriculumClass():
    def __init__(self, curr_type, t_pred, config, N_epochs=None):
        self.curr_type = curr_type
        self.t_pred = t_pred
        self.config = config
        self.N_epochs = N_epochs
        self.previous_loss = 0
        self.ema_loss = 0
        self.N_history = 10
        self.loss_history = torch.zeros(self.N_history)
        self._idx = 0


        if self.curr_type is not None:
            self._is_curriculum_available(self.curr_type)
            self._update_curriculum = getattr(self, self.curr_type)
        else:
            self._update_curriculum = self._pass
        
        """
        if curriculum_config is not None:
            self.curr_type = curriculum_config['type']
            self.curriculum_config = curriculum_config
            if N_epochs is not None and 'N_epochs' not in curriculum_config:
                self.curriculum_config['N_epochs'] = N_epochs
            self.curriculum_state = {}

            self._is_curriculum_available(self.curr_type)
            self._update_curriculum = getattr(self, self.curr_type)
        else:
            self.curr_type = None
            self.update_curriculum = self._pass 
        """
    
    def _is_curriculum_available(self, curr_type):
        if curr_type not in dir(self):
            curr_types = [
                attr for attr in dir(CurriculumClass)\
                    if attr[0] != '_' and callable(getattr(self, attr))
            ]
            raise ValueError(f"Cannot evaluate {curr_type}, either add a new function to the Metrics class or use one of the following:\n\t{curr_types}")

    def update_curriculum(self, epoch, loss):
        self.loss = loss
        
        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = 0.9*self.ema_loss + 0.1*loss

        self.loss_history[self._idx] = loss
        self._idx = (self._idx + 1) % self.N_history
        self.loss_std_ratio = torch.std(self.loss_history)/torch.abs(torch.mean(self.loss_history))

        return self._update_curriculum(epoch)
    
    def _pass(self, *args, **kwargs):
        pass

    def load_curriculum(self, curriculum_state):
        self.curriculum_state = curriculum_state
    
    def initialize_curriculum(self, epoch, model, train_loader, eval_loaders):
        if self.curr_type is not None:
            getattr(self, f"_{self.curr_type}_init")(
                epoch, model, train_loader, eval_loaders
            )
    
    def exponential(self, epoch):
        if self.config['metric'] == 'loss':
            update = self.loss < self.config['cut_off']
        elif self.config['metric'] == 'loss_std_ratio':
            update = self.loss_std_ratio < self.config['cut_off']
        else:
            raise ValueError(f"Cannot handle metric type {self.config['metric']}")

        if update:
            self.t_pred = self.t_pred*(1 + self.config['scale'])
        return self.t_pred

    def __exponential(self, epoch, **kwargs):
        # Smooth exponential progression
        progress = epoch / self.curriculum_config['N_epochs']
        exp_progress = 1 - np.exp(-3 * progress)  # Asymptotic to 1
        length = self.curriculum_config['init_length'] + (self.curriculum_config['final_length'] - self.curriculum_config['init_length']) * exp_progress
        t_pred = max(
            self.curriculum_config['init_length'],
            min(self.curriculum_config['final_length'], int(round(length)))
        )

        # Check if new prediction length will fit into gpu memory
        pass_memory, max_t_pred = self._memory_check(
            model.device, mem_unit, t_pred, mem_scale
        )
        if t_pred != self.t_pred and pass_memory:
            self.t_pred=t_pred
            train_loader.dataset.set_t_pred(t_pred)
            # CHANGING EVAL SET
            for label in eval_loaders.keys():
                eval_loaders[label].dataset.set_t_pred(t_pred)
            self.curriculum_state['t_pred'] = t_pred
            return True
        return False
    
    def _exponential_init(self, epoch, model, train_loader, eval_loaders):
        if len(self.curriculum_state) == 0:
            self.curriculum_state['last_change'] = 1
            self.exponential(epoch, model, train_loader, eval_loaders)
        else:
            self.t_pred = self.curriculum_state['t_pred']
            train_loader.dataset.set_t_pred(self.t_pred)
            # CHANGING EVAL SET
            for label in eval_loaders.keys():
                eval_loaders[label].dataset.set_t_pred(self.t_pred)


class DenseNet(nn.Module):
    def __init__(
            self,
            layers,
            dtype=torch.float64,
            output_activation=None,
            normalize=False
        ):
        super().__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.dtype=dtype
        self.activation = nn.GELU()
        #self.activation = nn.Tanh()
        #self.activation = nn.ELU()
        
        self.layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(nn.Linear(layers[i], layers[i+1], dtype=dtype, bias=False))
            if i != self.n_layers - 1:
                if normalize:
                    raise NotImplementedError
                    self.layers.append(nn.BatchNorm1d(layers[i + 1], dtype=dtype))
                self.layers.append(self.activation)

        if output_activation is not None:
            self.layers.append(nn.GELU())
    
    def forward(self, x):
        x = torch.movedim(x, 1, -1)
        for l in self.layers:
            x = l(x)
        x = torch.movedim(x, -1, 1)
        return x

class Trainer(CurriculumClass):
    def __init__(
            self,
            model,
            integrator_config,
            lr=1e-3,
            N_epochs=None,
            t_max=None,
            t_pred=0.1,
            t_init=0.0,
            curr_type=None,
            curr_config=None
        ) -> None:
        super().__init__(curr_type=curr_type, t_pred=t_pred, config=curr_config)
        self.dtype = model.dtype
        self.model = model

        assert N_epochs is not None or t_max is not None
        self.N_epochs = np.inf if N_epochs is None else N_epochs
        self.t_max = np.inf if t_max is None else t_max


        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        #self.loss_fxn = self._relative_MSE
        self.loss_fxn = self._relative_MSE

        self.integrator = tpd.RKParallelUniformAdaptiveStepsizeSolver(
            **integrator_config
        )

    @staticmethod
    def _MSE(pred, target):
        pred = torch.flatten(pred, start_dim=1)
        target = torch.flatten(target, start_dim=1)
        mse = torch.sum((pred - target)**2, dim=1, keepdim=True)
        return mse
    
    @staticmethod
    def _relative_MSE(pred, target):
        pred = torch.flatten(pred, start_dim=1)
        target = torch.flatten(target, start_dim=1)
        #print("diff", (pred - target))
        #print("pred", pred)
        #print("targ", target)
        return torch.sum(
            ((pred - target)/(torch.abs(target) + 1e-10))**2, #TODO: make eps smaller
            dim=1, keepdim=True
        )
    
    def _MSE_combined(self, pred, target):
        return self._MSE(pred, target) + self._relative_MSE(pred, target)

    def loss_integrad(self, t, model, verbose=False):
        jac = model(t)
        """
        jac = torch.autograd.functional.jacobian(
            lambda t: torch.sum(model(t), axis=0),
            t,
            create_graph=True,
            vectorize=True,
        ).transpose(0, 1)[:, :, 0]
        """
        
        #TODO: use guassian instead and t_init
        scale = torch.exp(-1*t)
        #scale /= 1.0 - torch.exp(-1*torch.tensor(self.t_pred))
        if verbose:
            print("pred", jac)
            print("targ", self.ode_fxn(t))
            #print("scale", scale)
        return self.loss_fxn(jac, self.ode_fxn(t))*scale
    

    def train(self, ode_fxn, initial_condition, t_init=0.0):
        self.ode_fxn = ode_fxn
        t_init = torch.tensor([t_init], requires_grad=True, dtype=self.dtype).unsqueeze(-1)
        self.model.train()
        self.model.compile()

        print("Optimizer Parameters:")
        for i, param_group in enumerate(self.optimizer.param_groups):
            print(f"\nParameter Group {i}:")
            for key, value in param_group.items():
                if key == 'params':
                    # 'params' contains a list of actual parameter tensors
                    print(f"  {key}: {len(value)} tensors")
                    for j, param in enumerate(value):
                        print(f"    Tensor {j} - Shape: {param.shape}, Requires Grad: {param.requires_grad}")
                        # You can also print the tensor values if desired (be mindful of large tensors)
                        # print(f"      Value: {param.data}")
                else:
                    print(f"  {key}: {value}")
        # Train t0
        loss_ratio, error_ratio, t0_count = 1, 1e10, 0
        print(self.model(t_init), self.ode_fxn(t_init))
        prev_loss = 1.0
        pred = torch.ones(1)
        print("Training initial conditions")
        while error_ratio > 1e-3:
            if t0_count % 100 == 0:
                print(f"T0 count {t0_count} | loss ratio: {loss_ratio} / error ratio: {error_ratio} / prediction: {torch.squeeze(pred).item()}")
            self.optimizer.zero_grad()
            pred = self.model(t_init) 
            loss = torch.mean(
                self.loss_fxn(pred, initial_condition)
            )
            #print("LOSS 0", loss.shape, loss)
            #print(self.model(t_init), self.ode_fxn(t_init))
            loss.backward()
            #print("BEFORE", self.model.layers[0].bias.data[:5])
            #print("GRAD", self.model.layers[0].bias.grad[:5]*1e5)
            self.optimizer.step()
            loss_ratio = torch.abs(prev_loss - loss)/prev_loss
            error_ratio = torch.abs(pred - initial_condition)/(torch.abs(initial_condition) + 1e-9)
            error_ratio = torch.squeeze(error_ratio)
            prev_loss = loss
            #print("AFTER", self.model.layers[0].bias.data[:5])
            #print(t0_count, self.t_pred, loss, loss_ratio)
            t0_count += 1
        print("Finished training initial conditions", loss_ratio, error_ratio)

        times, epoch_count = t_init, 0
        train_criteria = True
        while train_criteria:
            if epoch_count % 250 == 0:
                print(f"Epoch/Time {epoch_count}/{self.t_pred}: {loss.item()}")
                print("INIT", torch.squeeze(t_init).item(), torch.squeeze(self.model(t_init)).item())
                self.loss_integrad(torch.arange(10, dtype=self.dtype).unsqueeze(-1)*self.t_pred/9., self.model, verbose=True)
                self.loss_integrad(torch.arange(5, dtype=self.dtype).unsqueeze(-1)*5./4., self.model, verbose=True)
            self.optimizer.zero_grad()
            self.update_curriculum(epoch_count, loss)
            if times[-1] < self.t_pred:
                times = torch.concatenate(
                    [times, torch.tensor([self.t_pred]).unsqueeze(-1)], dim=0
                )
            times = torch.tensor([t_init, self.t_pred]).unsqueeze(-1)
            #print("TIMES", times.shape, loss, torch.std(self.loss_history), self.loss_history, times)
            integral_output = self.integrator.integrate(
                ode_fxn=self.loss_integrad, t=times, ode_args=(self.model,)
            )
            loss = integral_output.loss
            #print("BEFORE", self.model.layers[0].weight.data[:5])
            #print("OUTPUT", integral_output)
            if not integral_output.gradient_taken:
                #print("taking gradients")
                loss.backward()
            """
            
            t_eval = torch.arange(100).unsqueeze(1)*self.t_pred/99
            loss = torch.mean(
                self.loss_fxn(self.model(t_eval), self.ode_fxn(t_eval))
            )
            loss.backward()
            """
            # T0 loss
            init_loss = 10000*torch.mean(
                self.loss_fxn(self.model(t_init), initial_condition)
            )
            #init_loss.backward()
            #print("GRAD", self.model.layers[0].weight.grad[:5]*1e5)
            self.optimizer.step()
            #print("AFTER", self.model.layers[0].weight.data[:5])
            #times = integral_output.t_optimal.clone()
            #times = integral_output.t_optimal.detach()
            #times.requires_grad_(True)

            epoch_count += 1
            train_criteria = self.N_epochs is not None and epoch_count < self.N_epochs
            train_criteria *= self.t_max is not None and times[-1,0] < self.t_max


        

if __name__ == "__main__":
    # Get ODE
    #ode_fxn = linear()
    ode_fxn = quadratic()

    # Get Model
    model = DenseNet([1, 32, 64, 32, ode_fxn.N_dims])

    # Get Integrator
    
    #integral_output = parallel_integrator.integrate(
    #    ode, t_init=t_init, t_final=t_final
    #)
    
    # Setup Trainer
    trainer = Trainer(
        model=model,
        integrator_config={
            'method' : 'dopri5',
            'atol' : 1e-5,
            'rtol' : 1e-3
        },
        curr_type='exponential',
        curr_config={'metric' : 'loss', 'cut_off' : 1e-4, 'scale' : 0.05},
        t_pred=1e-3,
        t_max=100,
        #t_init_lr=1e-10,
        lr=1e-5
    )

    # Solve ODE
    trainer.train(ode_fxn, initial_condition=torch.zeros((1,1)))

    # Evaluate Solver

