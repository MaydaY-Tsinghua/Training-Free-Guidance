from utils.configs import Arguments
import torch
from torchvision import transforms

from UniDet_eval.numeracy_eval import load_expert_model, score_with_prompt_and_pred
from UniDet_eval.spatial_2D_eval import get_spatial_score_from_prompt_and_pred


class UniDetGuider:
    def __init__(self, args: Arguments):
        self.args = args
        if args.dataset == 'numeracy':
            self.model, self.transform = load_expert_model(task='obj_detection', ckpt="R50")
        elif args.dataset == 'spatial':
            self.model, self.transform = load_expert_model(task='obj_detection', ckpt="RS200")
        
        self.model = self.model.to(args.device).half().eval()

    def set_prompt(self, prompt):
        self.prompt = prompt
    
    def get_guidance(self, x, func=lambda x: x, post_process=lambda x: x, return_logp=True, **kwargs):
        if self.args.dataset == 'numeracy':
            return self.get_numeracy(x, func, post_process, return_logp, **kwargs)
        elif self.args.dataset == 'spatial':
            return self.get_spatial(x, func, post_process, return_logp, **kwargs)
        
    
    def get_numeracy(self, x, func=lambda x: x, post_process=lambda x: x, return_logp=True, **kwargs):
        assert return_logp, "UniDet can not take grad"
        x = func(x)
        x = post_process(x)
        x = x * 255
        x = self.transform(x)
        x = x.flip(dims=[1])   ## reverse channels as in Line 42 in https://github.com/Karine-Huang/T2I-CompBench/blob/main/UniDet_eval/experts/obj_detection/generate_dataset.py 
        x = x.to(self.args.device)
        test_data = []
        for image in x:
            test_data.append({'image': image})
        with torch.no_grad():
            test_pred = self.model(test_data)
        score_map = score_with_prompt_and_pred(self.prompt, test_pred)
        score_map = torch.tensor(score_map).to(self.args.device)
        logprob = torch.log(score_map + 1e-6)
        return logprob
    
    def get_spatial(self, x, func=lambda x: x, post_process=lambda x: x, return_logp=True, **kwargs):
        assert return_logp, "UniDet can not take grad"
        x = func(x)
        x = post_process(x)
        x = x * 255
        x = self.transform(x)
        x = x.flip(dims=[1])
        x = x.to(self.args.device)
        test_data = []
        for image in x:
            test_data.append({'image': image})

        with torch.no_grad():
            test_pred = self.model(test_data)
        
        score_map = get_spatial_score_from_prompt_and_pred(self.prompt, test_pred)
        score_map = torch.tensor(score_map).to(self.args.device)
        logprob = torch.log(score_map + 1e-6)
        return logprob