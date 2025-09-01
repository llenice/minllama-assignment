from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")


                # State should be stored in this dictionary
                # state存放该参数跨 step 持久的变量
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["moment_1"] = torch.zeros_like(p.data)
                    state["moment_2"] = torch.zeros_like(p.data)

                moment_1, moment_2 = state["moment_1"], state["moment_2"]
                state["step"] += 1
                t = state["step"]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]
                
                # Update first and second moments of the gradients
                # moment_1 = beta1 * moment_1 + (1 - beta1) * grad
                # moment_2 = beta2 * moment_2 + (1 - beta2) * (grad ** 2)

                # （原地操作
                moment_1.mul_(beta1).add_(grad, alpha = 1 - beta1)
                moment_2.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                if correct_bias:
                    beta1_t = beta1 ** t
                    beta2_t = beta2 ** t
                    m_hat = moment_1 / (1 - beta1_t)
                    v_hat = moment_2 / (1 - beta2_t)

                # Update parameters
                # θ_t = θ_{t-1} - α * m_hat_t / (√v_hat_t + ε)
                p.data = p.data - alpha * m_hat / (v_hat.sqrt() + eps)

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                # w = w - λ·lr·w
                if weight_decay:
                    p.data.add_(p.data, alpha = - weight_decay * alpha)

        return loss