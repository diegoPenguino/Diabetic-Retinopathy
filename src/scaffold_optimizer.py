from torch.optim import Optimizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class Scaffold_Optimizer(Optimizer):
    def __init__(self, params, lr, weight_decay=1e-5):
        defaults = dict(lr=lr)
        super(Scaffold_Optimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, closure=None):

        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            for p, c, ci in zip(
                group["params"], server_controls.values(), client_controls.values()
            ):
                if p.grad is None:
                    continue
                ## IMPLEMENT ADAM TO THIS CRAP
                dp = p.grad.data + c.data.to(device) - ci.data.to(device)
                p.data = p.data - dp.data * group["lr"]

        return loss
