from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer

#PC grad implementation
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Collect gradients for each parameter tensor
        task_gradients = []

        for group in self.param_groups:
            param_group_grads = []
            for p in group['params']:
                if p.grad is None:
                    param_group_grads.append(None)
                else:
                    if p.grad.data.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    param_group_grads.append(p.grad.data.clone())
            task_gradients.append(param_group_grads)

        # Apply gradient surgery for each parameter tensor
        for param_grads in zip(*task_gradients):
            for i, g_i in enumerate(param_grads):
                if g_i is None:
                    continue
                for j, g_j in enumerate(param_grads):
                    if i != j and g_j is not None:
                        g_dot = torch.dot(g_i.view(-1), g_j.view(-1))
                        if g_dot > 0:
                            g_i -= g_dot / g_j.norm().item() ** 2 * g_j

        # Update parameters with the modified gradients
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = task_gradients[group_idx][param_idx]

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss



#naive gradient surgery 
class AdamW_old(Optimizer):
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

    def gradient_surgery(self, grads):
        """
        Apply gradient surgery to minimize gradient interference between tasks.
        This implementation ensures that we handle each parameter's gradient separately.
        """
        for i in range(len(grads)):
            for j in range(i + 1, len(grads)):
                g_i, g_j = grads[i], grads[j]
                if g_i is not None and g_j is not None and g_i.shape == g_j.shape:
                    g_dot = torch.dot(g_i.flatten(), g_j.flatten())
                    if g_dot > 0:
                        g_i -= g_dot / g_j.norm().item() ** 2 * g_j

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        grads = []
        for group in self.param_groups:
            # Collect gradients
            for p in group["params"]:
                if p.grad is None:
                    grads.append(None)
                else:
                    grads.append(p.grad.data.clone())

        # Apply gradient surgery
        self.gradient_surgery([g for g in grads if g is not None])

        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                if grads[idx] is None:
                    continue
                grad = grads[idx]
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                state = self.state[p]
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                lam = group["weight_decay"]
                if len(state) == 0:
                    state["m"], state["v"] = torch.zeros_like(p), torch.zeros_like(p)
                    state["t"] = 0
                m, v, t = state["m"], state["v"], state["t"]
                t += 1
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.square(grad)
                alpha_t = alpha * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
                p.data = p.data - alpha_t * m / (torch.sqrt(v) + eps)
                p.data = p.data - alpha * lam * p
                state["m"], state["v"], state["t"] = m, v, t
        return loss




'''from typing import Callable, Iterable, Tuple
import math

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

                # State should be stored in this dictionary.
                state = self.state[p]
    
                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # To complete this implementation:
                # 1. Update the first and second moments of the gradients.
                # 2. Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3. Update parameters (p.data).
                # 4. Apply weight decay after the main gradient-based updates.
                # Refer to the default project handout for more details.

                # Fetch hyperparams
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                lam = group["weight_decay"]
                
                # Initialize first and second moment if not initialized
                if len(state) == 0:
                    state["m"], state["v"] = torch.zeros_like(p), torch.zeros_like(p)
                    state["t"] = 0

                m, v, t = state["m"], state["v"], state["t"]
                t += 1
                # Load first and second moment
                m = beta1 * m + (1 - beta1) * grad #First moment
                v = beta2 * v + (1 - beta2) * torch.square(grad) # Second moment
                alpha_t = alpha * math.sqrt(1 - math.pow(beta2, t)) / (1 -  math.pow(beta1, t)) #Bias correction
                p.data = p.data - alpha_t * m / (torch.sqrt(v) + eps) # Step 3
                p.data = p.data - alpha * lam * p # weight decay

                # Write into state
                state["m"], state["v"], state["t"] = m, v, t
        return loss
'''