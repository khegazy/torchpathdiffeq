from __future__ import annotations

import numpy as np
import torch


class CurriculumClass:
    def __init__(
        self, curr_type, t_pred, t_max, config, dtype=torch.float64, N_epochs=None
    ):
        self.curr_type = curr_type
        self.config = config
        self.N_epochs = N_epochs
        self.previous_loss = 0
        self.ema_loss = 0
        self.N_history = 10
        self.loss_history = torch.zeros(self.N_history)
        self._idx = 0
        self.dtype = dtype

        self.prev_t_pred = 1e-9
        self.t_pred = torch.tensor(t_pred, dtype=self.dtype)
        self.t_max = torch.tensor(t_max, dtype=self.dtype)
        if self.curr_type is not None:
            self._is_curriculum_available(self.curr_type)
            self._update_curriculum = getattr(self, self.curr_type)
        else:
            self._update_curriculum = self._pass

        self.curr_patience = None
        if "cutoff_patience" in self.config:
            assert "cutoff" in self.config
            if "cutoff_scale" not in self.config:
                self.config["cutoff_scale"] = 1.2
            assert self.config["cutoff_scale"] > 1.0
            self.curr_patience = 0

            self.config["default_cutoff"] = self.config["cutoff"]

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
                attr
                for attr in dir(CurriculumClass)
                if attr[0] != "_" and callable(getattr(self, attr))
            ]
            raise ValueError(
                f"Cannot evaluate {curr_type}, either add a new function to the Metrics class or use one of the following:\n\t{curr_types}"
            )

    # def _calculate_filter(self):

    def update_curriculum(self, epoch, loss, force_update=False):
        self.loss = loss

        if self.ema_loss is None:
            self.ema_loss = loss
        else:
            self.ema_loss = 0.9 * self.ema_loss + 0.1 * loss

        self.loss_history[self._idx] = loss
        self._idx = (self._idx + 1) % self.N_history
        self.loss_std_ratio = torch.std(self.loss_history) / torch.abs(
            torch.mean(self.loss_history)
        )

        if self.t_pred != self.t_max:
            # Increase the cutoff is training stagnates and t_max has not been reached
            if self.curr_patience is not None:
                if self.loss > self.config["cutoff"]:
                    self.curr_patience += 1
                    if self.curr_patience >= self.config["cutoff_patience"]:
                        self.config["cutoff"] = (
                            self.config["cutoff"] * self.config["cutoff_scale"]
                        )
                        print(
                            f"Hit curriculum patience limit, increasing cutoff to {self.config['cutoff']}"
                        )
                        self.curr_patience = 0
                else:
                    # if self.curr_patience < self.config['cutoff_patience']/2:
                    #    self.config['cutoff'] = self.config['cutoff']/self.config['cutoff_scale']
                    self.config["cutoff"] = self.config["default_cutoff"]
                    self.curr_patience = 0
            if force_update:
                t_update = self.t_pred + self.config["scale"]
                updated_t = True
            else:
                t_update, updated_t = self._update_curriculum(epoch)
            t_update = np.minimum(t_update, self.t_max)
            self.prev_t_pred = self.t_pred
            self.t_pred = t_update
            return updated_t and t_update < self.t_max
        return None

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
        if self.config["metric"] == "loss":
            update = self.loss < self.config["cutoff"]
        elif self.config["metric"] == "loss_std_ratio":
            update = self.loss_std_ratio < self.config["cutoff"]
        else:
            raise ValueError(f"Cannot handle metric type {self.config['metric']}")

        if update:
            return self.t_pred + self.config["scale"], True
            # return self.t_pred*(1 + self.config['scale']), True #TODO THIS IS EXPONENTIAL
        return self.t_pred, False

    def __exponential(self, epoch, **kwargs):
        # Smooth exponential progression
        progress = epoch / self.curriculum_config["N_epochs"]
        exp_progress = 1 - np.exp(-3 * progress)  # Asymptotic to 1
        length = (
            self.curriculum_config["init_length"]
            + (
                self.curriculum_config["final_length"]
                - self.curriculum_config["init_length"]
            )
            * exp_progress
        )
        t_pred = max(
            self.curriculum_config["init_length"],
            min(self.curriculum_config["final_length"], int(round(length))),
        )

        # Check if new prediction length will fit into gpu memory
        pass_memory, max_t_pred = self._memory_check(
            model.device, mem_unit, t_pred, mem_scale
        )
        if t_pred != self.t_pred and pass_memory:
            self.t_pred = t_pred
            train_loader.dataset.set_t_pred(t_pred)
            # CHANGING EVAL SET
            for label in eval_loaders:
                eval_loaders[label].dataset.set_t_pred(t_pred)
            self.curriculum_state["t_pred"] = t_pred
            return True
        return False

    def _exponential_init(self, epoch, model, train_loader, eval_loaders):
        if len(self.curriculum_state) == 0:
            self.curriculum_state["last_change"] = 1
            self.exponential(epoch, model, train_loader, eval_loaders)
        else:
            self.t_pred = self.curriculum_state["t_pred"]
            train_loader.dataset.set_t_pred(self.t_pred)
            # CHANGING EVAL SET
            for label in eval_loaders:
                eval_loaders[label].dataset.set_t_pred(self.t_pred)
