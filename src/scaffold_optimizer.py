from torch.optim import Optimizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class Scaffold_Optimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(Scaffold_Optimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls):
        for group in self.param_groups:
            for p, c, ci in zip(
                group["params"], server_controls.values(), client_controls.values()
            ):
                if p.grad is None:
                    continue
                dp = p.grad.data + c.data.to(device) - ci.data.to(device)
                p.data = p.data - dp.data * group["lr"]
