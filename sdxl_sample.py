from diffusers import StableDiffusionXLPipeline,AutoencoderTiny,DDIMScheduler
import torch
pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B",
                                                 torch_dtype=torch.float16,variant="fp16",
                                                 use_safetensors=True
                                                 )
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl",
                                            torch_dtype=torch.float16
                                            )
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config,timestep_spacing="trailing")
pipe.to("cuda")
generator = torch.manual_seed(3)
unet = pipe.unet.eval()

from ptflops import get_model_complexity_info
macs, params = get_model_complexity_info(unet, (4, 128, 128))
print("UNet MACs:", macs)