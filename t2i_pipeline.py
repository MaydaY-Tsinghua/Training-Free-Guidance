from diffusion.stable_diffusion import StableDiffusionSampler
from methods.base import BaseGuidance
from utils.configs import Arguments
from TFG_args import args
import os
from evaluations.blip_vqa import BLIPEvaluator
from tqdm import tqdm

class T2IPipeline:
    def __init__(self, args: Arguments, sampler: StableDiffusionSampler=None, guider: BaseGuidance=None):
        self.sampler = StableDiffusionSampler(args) if sampler is None else sampler
        self.guider = BaseGuidance(args) if guider is None else guider
        self.prompts = self._prepare_prompts(args)
        self._setup_evaluator(args)
        self.args = args
    
    def _setup_evaluator(self, args):
        if args.dataset == 'color' or args.dataset == 'shape' or args.dataset == 'texture':
            self.evaluator = BLIPEvaluator(args)
        if args.dataset == 'spatial' or args.dataset == 'numeracy' or args.dataset == 'complex':
            from evaluations.uniDet import UniDeTEvaluator
            self.evaluator = UniDeTEvaluator(args)
        if args.dataset == 'non-spatial':
            from evaluations.clip_eval import CLIPEvaluator
            self.evaluator = CLIPEvaluator(args)



    def _prepare_prompts(self, args):
        
        if args.dataset == 'color':
            prompts = [line.strip() for line in open("T2I-CompBench/examples/dataset/color_val.txt", 'r').readlines()]
        elif args.dataset == 'numeracy':
            prompts = [line.strip() for line in open("T2I-CompBench/examples/dataset/numeracy_val.txt", 'r').readlines()]
        elif args.dataset == 'shape':
            prompts = [line.strip() for line in open("T2I-CompBench/examples/dataset/shape_val.txt", 'r').readlines()]
        elif args.dataset == 'spatial':
            prompts = [line.strip() for line in open("T2I-CompBench/examples/dataset/spatial_val.txt", 'r').readlines()]
        elif args.dataset == 'texture':
            prompts = [line.strip() for line in open("T2I-CompBench/examples/dataset/texture_val.txt", 'r').readlines()]
        elif args.dataset == 'complex':
            prompts = [line.strip() for line in open("T2I-CompBench/examples/dataset/complex_val.txt", 'r').readlines()]
        elif args.dataset == 'non-spatial':
            prompts = [line.strip() for line in open("T2I-CompBench/examples/dataset/non_spatial_val.txt", 'r').readlines()]
        else:
            prompts = ["flower"] * args.num_samples
        
        return prompts


    def sample(self, num_samples: int=None):
        os.makedirs(os.path.join(self.args.logging_dir, 'samples'), exist_ok=True)
        if num_samples is None:
            num_samples = len(self.prompts)
        else:
            num_samples = min(num_samples, len(self.prompts))

        total_results = {}
        total_results['prompt'] = []
        for i in tqdm(range(num_samples), total=num_samples):
            prompt = self.prompts[i]
            self.guider.set_prompt(prompt=prompt)
            sample, extra_results = self.sampler.sample(prompt, self.guider)
            sample = self.sampler.tensor_to_obj(sample)
            for id, image in enumerate(sample):
                file_name = f"{prompt}_{id:06d}.png"
                image.save(os.path.join(self.args.logging_dir,'samples', file_name))
            for key, value in extra_results.items():
                if key not in total_results:
                    total_results[key] = []
                total_results[key].append(value)
            total_results['prompt'].append(prompt)
        import pickle
        with open(os.path.join(self.args.logging_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(total_results, f)
        score = self.evaluator.evaluate(self.args.logging_dir)
        for key, value in total_results.items():
            if key not in ['prompt', 'score_list']:
                total_results[key] = sum(total_results[key]) / len(total_results[key])
        print(f"Evaluation score: {score}")
        return score, total_results

        
        
        