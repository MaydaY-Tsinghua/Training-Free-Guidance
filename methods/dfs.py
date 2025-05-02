from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor

import math
from torch.autograd import grad
import torch
from functools import partial

from tasks.utils import rescale_grad


class DFSGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.device = args.device

    def get_threshold(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        # scheduler = alpha_prod_ts / alpha_prod_t_prevs

        # return self.args.threshold * scheduler[t] * len(scheduler) / scheduler.sum()
        # scheduler = {25: 0.8, 35: 0.8, 45: 0.9}
        # return scheduler[t] if t in scheduler else self.args.threshold
        return self.args.threshold
    
    def reset(self, **kwargs):
        self.budget = self.args.budget
        self.buffer = [{} for _ in range(self.args.inference_steps)]
        assert self.args.per_sample_batch_size == 1
    
    def get_resampling_steps(self, **kwargs):
        return list(range(self.args.start_step, self.args.inference_steps, self.args.step_size))
    
                

    def guide_step(
        self,
        x: torch.Tensor,
        t: int,
        unet: torch.nn.Module,
        ts: torch.LongTensor,
        alpha_prod_ts: torch.Tensor,
        alpha_prod_t_prevs: torch.Tensor,
        eta: float,
        **kwargs,
    ) -> torch.Tensor:
        alpha_prod_t = alpha_prod_ts[t]
        alpha_prod_t_prev = alpha_prod_t_prevs[t]

        
        threshold = self.get_threshold(t, alpha_prod_ts, alpha_prod_t_prevs)
        # threshold = self.args.threshold
        i = t
        t = ts[t]   # convert from int space to tensor space


        # predict x_{t-1} using S(zt, hat_epsilon, t), this is also DDIM sampling
        x0 = self._predict_x0(x, unet(x, t), alpha_prod_t, **kwargs)
        
        
        accept = True
        if i in self.get_resampling_steps():
            logprob =self.guider.get_guidance(x0, return_logp=True, **kwargs)
            score = torch.exp(logprob)
            self.buffer[i].update({score: (x, x0)})
            if score < threshold and self.budget > 0:
                self.budget -= 1
                accept = False
            elif score > threshold:
                accept = True
            elif score < threshold and self.budget <= 0:
                accept = True
                x, x0 = self.buffer[i][max(self.buffer[i].keys())]
        

        if accept:
            x_prev = self._predict_x_prev_from_zero(
            x, x0, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)
            return x_prev, i+1

        else:
            next_noise_level = max(0, i - self.args.recur_depth)
            if next_noise_level == 0:
                x = torch.randn_like(x)
                return x, next_noise_level
            
            alpha_prod_t = alpha_prod_ts[next_noise_level]
            alpha_prod_t_prev = alpha_prod_ts[i]
            x = self._predict_xt(x, alpha_prod_t, alpha_prod_t_prev, **kwargs)
            return x, next_noise_level
