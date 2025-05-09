import math
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from tqdm import tqdm
from typing import List, Union, Optional, Dict, Any, Tuple, Callable

from diffusers.utils.torch_utils import randn_tensor

from utils.configs import Arguments
from .base import BaseSampler
from methods.base import BaseGuidance
from diffusers import StableDiffusionPipeline,DDIMScheduler,StableDiffusionXLPipeline,AutoencoderTiny
from  diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg
from utils.env_utils import *


def reshape(x, num_samples):
    if isinstance(x, torch.Tensor):
        x_uncond, x_cond = x.chunk(2)
        x_uncond = x_uncond[:num_samples]
        x_cond = x_cond[:num_samples]
        x = torch.cat([x_uncond, x_cond], dim=0)
    elif isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = reshape(v, num_samples)
    return x
    
        


class StableDiffusionSampler(BaseSampler):

    def __init__(self, args: Arguments):

        super(StableDiffusionSampler, self).__init__(args)
        self.image_size = args.image_size
        self.inference_steps = args.inference_steps
        self.eta = args.eta
        self.log_traj = args.log_traj
        self.generator = torch.manual_seed(self.seed)

        # FIXME: need to send batch_id to guider
        self.args = args
        # prepare unet, prev_t, alpha_prod, alpha_prod_prev...
        self._build_diffusion(args)

    @torch.no_grad()
    def decode(self, latents):
        return self.sd_pipeline.vae.decode(latents / self.sd_pipeline.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0]

    @torch.no_grad()
    def _build_diffusion(self, args):
        
        '''
            Different diffusion models should be registered here
        '''
        self.sd_pipeline = StableDiffusionXLPipeline.from_pretrained(args.model_name_or_path,
                                                                     torch_dtype=torch.float16,variant="fp16"
                                                                     ).to(self.device)
        self.scheduler = DDIMScheduler.from_config(self.sd_pipeline.scheduler.config, timestep_spacing="trailing")
        self.sd_pipeline.scheduler = self.scheduler
        unet = self.sd_pipeline.unet
        unet.eval()

        for param in unet.parameters():
            param.requires_grad = False

        self.scheduler.set_timesteps(args.inference_steps)
        ts = self.scheduler.timesteps

        alpha_prod_ts = self.scheduler.alphas_cumprod[ts]
        alpha_prod_t_prevs = torch.cat([alpha_prod_ts[1:], torch.ones(1) * self.scheduler.final_alpha_cumprod])

        self.height = self.width = self.sd_pipeline.unet.config.sample_size * self.sd_pipeline.vae_scale_factor

        # prepare prompts: str or List[str]
        # self.prompts = self._prepare_prompts(args)  

        # FIXME: classifier-free guidance params
        self.do_classifier_free_guidance = True
        self.guidance_scale = 5.0

        # check inputs. Raise error if not correct
        # self.sd_pipeline.check_inputs(prompt=self.prompts,prompt_2=None, height=self.height, width=self.width, callback_steps=None)

        self.unet, self.ts, self.alpha_prod_ts, self.alpha_prod_t_prevs = unet, ts, alpha_prod_ts, alpha_prod_t_prevs
        self.sd_pipeline.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl",
                                                               torch_dtype=torch.float16
                                                               ).to(self.device)
    
    # def _prepare_prompts(self, args):
        
    #     if args.dataset == 'parti_prompts':
    #         prompts = [line.strip() for line in open(PARTIPROMPOTS_PATH, 'r').readlines()][:args.num_samples]
    #     else:
    #         prompts = ["flower"] * args.num_samples
        
    #     return prompts

    

    @torch.no_grad()
    def sample(self, prompt: str,  guidance: BaseGuidance):
        tot_samples = []
        total_compute = 0
        total_score = []
        for sample_id in range(self.args.num_samples):
            self.args.batch_id = sample_id

            prompts = [prompt] * self.args.per_sample_batch_size

            # encode input prompts
            prompt_embeds, added_cond_kwargs = self.prepare_unet_kwargs(prompt=prompts)

            latents = self.sd_pipeline.prepare_latents(
                len(prompts),
                self.sd_pipeline.unet.config.in_channels,
                self.height,
                self.width,
                prompt_embeds.dtype,
                self.device,
                generator=self.generator
            )

            # for t in range(self.inference_steps):
            t = 0
            guidance.reset()
            while t < self.inference_steps:
                @torch.no_grad()
                def stable_diffusion_unet(latents, t):
                    
                    num_samples = latents.shape[0]

                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    # from torch.profiler import profile
                    # with profile(
                    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    #     with_flops=True,
                    # ) as prof:
                    noise_pred = self.unet(latent_model_input, 
                                        t, 
                                        encoder_hidden_states=reshape(prompt_embeds, num_samples),
                                        added_cond_kwargs=reshape(added_cond_kwargs, num_samples)
                                        )[0]
                    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        
                    return noise_pred

                total_compute += latents.shape[0]
                latents, t = guidance.guide_step(
                    latents, t, stable_diffusion_unet,
                    self.ts,
                    self.alpha_prod_ts, 
                    self.alpha_prod_t_prevs,
                    self.eta
                )

            logprob = guidance.guider.get_guidance(latents, return_logp=True)
            score = torch.exp(logprob).item()
            total_score.append(score)
            image = self.decode(latents)
            tot_samples.append(image.clone().cpu())

        avg_compute = total_compute / self.args.num_samples
        avg_score = sum(total_score) / len(total_score)
        std_score = torch.std(torch.tensor(total_score)).item()
        return torch.concat(tot_samples), {'compute': avg_compute, 'score': avg_score, 'std_score': std_score, 'score_list': total_score}
        
    def tensor_to_obj(self, x):

        images = (x / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        
        return pil_images
    
    def obj_to_tensor(self, objs: List[Image.Image]) -> torch.Tensor:
        '''
            convert a list of PIL images into tensors
        '''
        images = [to_tensor(pil_image) for pil_image in objs]
        tensor_images = torch.stack(images).to(self.device)
        return tensor_images * 2 - 1
    





    @torch.no_grad()
    def prepare_unet_kwargs(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = "ugly, blurry, poor quality",
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        ):
        height = height or self.sd_pipeline.default_sample_size * self.sd_pipeline.vae_scale_factor
        width = width or self.sd_pipeline.default_sample_size * self.sd_pipeline.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.sd_pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device=self.device,
            num_images_per_prompt=1,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        add_text_embeds = pooled_prompt_embeds
        if self.sd_pipeline.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.sd_pipeline.text_encoder_2.config.projection_dim

        add_time_ids = self.sd_pipeline._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self.sd_pipeline._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device).repeat(batch_size * num_images_per_prompt, 1)
        added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
        return prompt_embeds, added_cond_kwargs
