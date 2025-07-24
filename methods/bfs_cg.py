from .base import BaseGuidance
from diffusers.utils.torch_utils import randn_tensor

import math
from torch.autograd import grad
import torch
from functools import partial

from tasks.utils import rescale_grad


class BFSCGGuidance(BaseGuidance):

    def __init__(self, args, **kwargs):
        super(BFSCGGuidance, self).__init__(args, **kwargs)
        self.device = args.device

    @torch.enable_grad()
    def tilde_get_guidance(self, x0, mc_eps, return_logp=False, **kwargs):

        # flat_x0 = (x0[None] + mc_eps) #.reshape(-1, *x0.shape[1:])
        # v_func = torch.vmap(partial(self.guider.get_guidance,
        #                             return_logp=True, 
        #                             check_grad=False,
        #                             **kwargs))
        # outs = v_func(flat_x0)
        
        # avg_logprobs = torch.logsumexp(outs, dim=0) - math.log(mc_eps.shape[0])
        
        flat_x0 = (x0[None] + mc_eps).reshape(-1, *x0.shape[1:])
        outs = self.guider.get_guidance(flat_x0, return_logp=True, check_grad=False, **kwargs)

        avg_logprobs = torch.logsumexp(outs.reshape(mc_eps.shape[0], x0.shape[0]), dim=0) - math.log(mc_eps.shape[0])
        
        if return_logp:
            return avg_logprobs

        _grad = torch.autograd.grad(avg_logprobs.sum(), x0)[0]
        _grad = rescale_grad(_grad, clip_scale=self.args.clip_scale, **kwargs)
        return _grad
    
    def get_noise(self, std, shape, eps_bsz=4, **kwargs):
        if std == 0.0:
            return torch.zeros((1, *shape), device=self.device)
        return torch.stack([self.noise_fn(torch.zeros(shape, device=self.device), std, **kwargs) for _ in range(eps_bsz)]) 
    # randn_tensor((4, *shape), device=self.device, generator=self.generator) * std
    
    def get_rho(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.rho_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.rho_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.rho * scheduler[t] * len(scheduler) / scheduler.sum()

    def get_mu(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.mu_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.mu_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.mu *  scheduler[t] * len(scheduler) / scheduler.sum()
    
    def get_std(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.sigma_schedule == 'decrease':    # beta_t
            scheduler = (1 - alpha_prod_ts) ** 0.5
        elif self.args.sigma_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)

        return self.args.sigma *  scheduler[t]
    
    def get_temp(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        scheduler = alpha_prod_ts / alpha_prod_t_prevs
        return self.args.temp * scheduler[t] * len(scheduler) / scheduler.sum()

    def resampling_steps(self, **kwargs):
        return list(range(self.args.start, self.args.inference_steps, self.args.step_size))

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
        bon_guider = kwargs.pop('bon_guider', None)
        rho = self.get_rho(t, alpha_prod_ts, alpha_prod_t_prevs)
        mu = self.get_mu(t, alpha_prod_ts, alpha_prod_t_prevs)
        std = self.get_std(t, alpha_prod_ts, alpha_prod_t_prevs)
        temp = self.get_temp(t, alpha_prod_ts, alpha_prod_t_prevs)
        i = t     # i is in index space
        t = ts[t]   # convert from int space to tensor space
        # alg 2 in classifier-guidance paper
        epsilon = unet(x, t)
        x_need_grad = x.clone().detach().requires_grad_(True)
        guidance = self.guider.get_guidance(x_need_grad, t, **kwargs)

        epsilon = epsilon - self.args.guidance_strength * guidance * ((1 - alpha_prod_t) ** (0.5))

        x_prev = self._predict_x_prev_from_eps(x, epsilon, alpha_prod_t, alpha_prod_t_prev, eta, t, **kwargs)
        x0 = self._predict_x0(x, epsilon, alpha_prod_t, **kwargs)

        if i in self.resampling_steps() and temp > 0:
            if bon_guider:
                logprobs = bon_guider.guider.get_guidance(x0, return_logp=True, check_grad=False, **kwargs)
            else:
                logprobs = self.guider.get_guidance(x0, return_logp=True, check_grad=False, **kwargs)
            num_children = x.shape[0] * torch.softmax(logprobs * temp, dim=0)
            num_children = torch.round(num_children).long()
            ## rebase
            if self.args.guidance_name == 'bfs-resample':
                resampled_indices = torch.repeat_interleave(
                    torch.arange(x.shape[0], device=x.device), num_children
                )[:x.shape[0]]
            ## pruning
            elif self.args.guidance_name == 'bfs-prune':
                resampled_indices = torch.where(num_children > 0)[0]
            x_prev = x_prev[resampled_indices]


        if i == len(ts) - 1:
            if bon_guider:
                logprobs = bon_guider.guider.get_guidance(x_prev, return_logp=True, check_grad=False, **kwargs)
            else:
                logprobs = self.guider.get_guidance(x_prev, return_logp=True, check_grad=False, **kwargs)
            x_prev = x_prev[torch.argmax(logprobs, dim=0)].unsqueeze(0)



        return x_prev
