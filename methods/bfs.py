from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor

import math
from torch.autograd import grad
import torch
from functools import partial

from tasks.utils import rescale_grad


class BFSGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.device = args.device

    
    def get_temp(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.rho_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.temp * scheduler[t] * len(scheduler) / scheduler.sum()

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

        
        temp = self.get_temp(t, alpha_prod_ts, alpha_prod_t_prevs)
        eta = 1.0   ## for random transition kernal in BFS resampling
        i = t
        t = ts[t]   # convert from int space to tensor space


        # predict x_{t-1} using S(zt, hat_epsilon, t), this is also DDIM sampling
        x0 = self._predict_x0(x, unet(x, t), alpha_prod_t, **kwargs)
        x_prev = self._predict_x_prev_from_zero(
            x, x0, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)
        
        if i in self.get_resampling_steps() and temp > 0:
            logprob = self.guider.get_guidance(x0, return_logp=True, **kwargs)
            logprob = logprob * temp
            bs = logprob.shape[0]
            prob = torch.softmax(logprob, dim=0)
            num_children = prob * bs
            # REBASE
            num_children = torch.round(num_children)
            resampled_indices = torch.repeat_interleave(torch.arange(bs,device=x.device), num_children.long(), dim=0)[:bs]
            # pruning
            # resampled_indices = torch.where(num_children > 0.5)[0]
            # # Beam Search Pruning
            # sorted_indices = torch.argsort(logprob, descending=True)
            # resampled_indices = sorted_indices[:-1] if len(sorted_indices) > 1 else sorted_indices
            x_prev = x_prev[resampled_indices]







        if i == len(ts) - 1:
            logprob = self.guider.get_guidance(x0, return_logp=True, **kwargs)
            x_prev = x_prev[torch.argmax(logprob)].unsqueeze(0)

        
        
        return x_prev, i+1