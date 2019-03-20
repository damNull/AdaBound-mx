# author: damNull
# date: 2019/3/18

import mxnet as mx
import math
from mxnet import nd
from mxnet.optimizer import Optimizer
from mxnet.ndarray import NDArray

class AdaBound(Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), final_lr=0.1, gamma=1e-3,
                 eps=1e-8, weight_decay=0, amsbound=False, **kwargs):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= final_lr:
            raise ValueError("Invalid final learning rate: {}".format(final_lr))
        if not 0.0 <= gamma < 1.0:
            raise ValueError("Invalid gamma parameter: {}".format(gamma))
        self.group = dict(learning_rate=lr, betas=betas, final_lr=final_lr, gamma=gamma, eps=eps,
                        wd=weight_decay, amsbound=amsbound)
        super(AdaBound, self).__init__(learning_rate=self.group['learning_rate'], wd=self.group['wd'], **kwargs)

    def create_state(self, index, weight):
        """Creates a state to duplicate weight."""
        state = dict()
        state['exp_avg'] = nd.zeros(weight.shape, weight.context, weight.dtype)
        state['exp_avg_sq'] = nd.zeros(weight.shape, weight.context, weight.dtype)
        if self.group['amsbound']:
            state['max_exp_avg_sq'] = nd.zeros(weight.shape, weight.context, weight.dtype)
        return state

    def update(self, index, weight, grad, state):
        """Performs w += rescale_grad * grad."""
        assert (isinstance(weight, NDArray))
        assert (isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        t = self._index_update_count[index]
        grad *= self.rescale_grad

        amsbound = self.group['amsbound']

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsbound:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = self.group['betas']
        if self.group['wd'] != 0:
            grad = grad + (wd * weight)

        exp_avg[:] *= beta1
        exp_avg[:] += ((1 - beta1) * grad)

        exp_avg_sq[:] *= beta2
        exp_avg_sq[:] = exp_avg_sq + (1 - beta2) * grad * grad

        denom = nd.zeros(weight.shape, ctx=weight.context)
        if amsbound:
            # Maintains the maximum of all 2nd moment running avg. till now
            max_exp_avg_sq = nd.max(nd.stack(max_exp_avg_sq, exp_avg_sq), axis=0)
            # Use the max. for normalizing running avg. of gradient
            denom[:] = max_exp_avg_sq.sqrt() + self.group['eps']
        else:
            denom[:] = exp_avg_sq.sqrt() + self.group['eps']

        bias_correction1 = 1 - beta1 ** t
        bias_correction2 = 1 - beta2 ** t

        step_size = self.group['learning_rate'] * math.sqrt(bias_correction2) / bias_correction1

        final_lr = self.group['final_lr'] * self.group['learning_rate'] / lr
        lower_bound = final_lr * (1 - 1 / (self.group['gamma'] * t + 1))
        upper_bound = final_lr * (1 + 1 / (self.group['gamma'] * t))
        step_size = nd.full(denom.shape, step_size, ctx=weight.context)
        step_size[:] /= denom
        step_size = nd.clip(step_size, lower_bound, upper_bound)
        step_size[:] *= exp_avg

        weight[:] -= step_size

