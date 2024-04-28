from torch.optim import Optimizer


class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

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
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group["lr"]

        return loss