import ImageReward as RM
from utils.configs import Arguments
import torch
from torchvision import transforms


class ImageRewardGuider:
    def __init__(self, args):
        self.args = args
        self.reward_model = RM.load(device=args.device).to(dtype=torch.float16)
        self.transform = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),                
                                            transforms.CenterCrop(224),
                                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                            ])
        self.device = torch.device(args.device)
    
    def set_prompt(self, prompt):
        self.prompt = prompt
        self.text_input = self.reward_model.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        self.prompt_attention_mask = self.text_input['attention_mask']
        self.prompt_ids = self.text_input['input_ids']

    def get_guidance(self, x_need_grad, func=lambda x: x, post_process=lambda x: x, return_logp=True, **kwargs):
        x = func(x_need_grad)
        x = post_process(x)
        x = self.transform(x)
        
        # Get the reward score
        rewards = self.reward_model.score_gard(self.prompt_ids, self.prompt_attention_mask, x)
        rewards = rewards.squeeze(-1)

        if return_logp:
            return rewards
        
        # Compute the gradient
        grad = torch.autograd.grad(rewards.mean(), x_need_grad)[0]
        
        return grad


