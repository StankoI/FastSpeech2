import torch
import numpy as np


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling.

    Base schedule is Noam (linear warmup, then 1/sqrt(step) decay, with optional
    step-wise anneals). On top of that, an optional ReduceLROnPlateau-style factor
    lowers the LR further whenever the monitored (validation) loss stops improving.
    """

    def __init__(self, model, train_config, model_config, current_step):

        opt = train_config["optimizer"]
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            betas=opt["betas"],
            eps=opt["eps"],
            weight_decay=opt["weight_decay"],
        )
        self.n_warmup_steps = opt["warm_up_step"]
        self.anneal_steps = opt["anneal_steps"]
        self.anneal_rate = opt["anneal_rate"]
        self.current_step = current_step
        self.init_lr = np.power(model_config["transformer"]["encoder_hidden"], -0.5)

        # Loss-aware (plateau) reduction layered on top of the Noam schedule.
        self.plateau_enabled = opt.get("plateau_enabled", True)
        self.plateau_factor = opt.get("plateau_factor", 0.5)       # multiply LR by this on a plateau
        self.plateau_patience = opt.get("plateau_patience", 5)     # #evals w/o improvement before reducing
        self.plateau_min_delta = opt.get("plateau_min_delta", 0.0) # min loss drop counted as improvement
        self.plateau_min_scale = opt.get("plateau_min_scale", 0.01)  # floor on the plateau multiplier
        self._plateau_scale = 1.0
        self._best_loss = float("inf")
        self._num_bad_evals = 0

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def update_learning_rate(self):
        """Advance the schedule without stepping (used by AMP GradScaler)."""
        self._update_learning_rate()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def register_val_loss(self, loss):
        """Call after each validation. Reduces the LR multiplier when the loss
        has not improved for `plateau_patience` consecutive evaluations."""
        if not self.plateau_enabled:
            return self._plateau_scale
        if loss < self._best_loss - self.plateau_min_delta:
            self._best_loss = loss
            self._num_bad_evals = 0
        else:
            self._num_bad_evals += 1
            if self._num_bad_evals >= self.plateau_patience:
                self._plateau_scale = max(
                    self._plateau_scale * self.plateau_factor, self.plateau_min_scale
                )
                self._num_bad_evals = 0
        return self._plateau_scale

    def get_lr(self):
        """Current learning rate actually applied to the optimizer."""
        return self._optimizer.param_groups[0]["lr"]

    def get_plateau_state(self):
        return {
            "plateau_scale": self._plateau_scale,
            "best_loss": self._best_loss,
            "num_bad_evals": self._num_bad_evals,
        }

    def load_plateau_state(self, state):
        if not state:
            return
        self._plateau_scale = state.get("plateau_scale", 1.0)
        self._best_loss = state.get("best_loss", float("inf"))
        self._num_bad_evals = state.get("num_bad_evals", 0)

    def _get_lr_scale(self):
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale() * self._plateau_scale

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
