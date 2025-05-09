import torch
import clip
from utils.configs import Arguments
from torchvision import transforms

class CLIPGuider:
    def __init__(self, args: Arguments):
        self.args = args
        self.device = args.device
        self.model, preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def set_prompt(self, prompt):
        self.prompt = prompt
        self.text_features = self.model.encode_text(clip.tokenize(prompt).to(self.device))
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def get_guidance(self, x_need_grad, func=lambda x: x, post_process=lambda x: x, return_logp=False, check_grad=True, **kwargs):
        assert return_logp == True 
        x_need_grad = func(x_need_grad)
        x = post_process(x_need_grad)
        x = self.transforms(x)
        
        with torch.no_grad():
            image_features = self.model.encode_image(x)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Calculate the cosine similarity between the image and text features
            cosine_similarity = (image_features @ self.text_features.T).squeeze(-1)
        
        return cosine_similarity

        