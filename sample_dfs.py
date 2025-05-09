from t2i_pipeline import T2IPipeline
from TFG_args import args, get_config, dataset_to_task
from methods.dfs import DFSGuidance
from diffusion.stable_diffusion import StableDiffusionSampler
import os
from copy import deepcopy

def sample(dataset, device):
    args.guidance_name = 'dfs'
    # args.model_name_or_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    args.per_sample_batch_size = 1
    args.dataset = dataset
    args.task = dataset_to_task(dataset)
    args.device = device

    csv_path = f"results_{args.dataset}_SSD_dfs.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as csv_file:
            csv_file.write("compute, recur_depth, start, step_size, budget, threshold, score\n")
    for start in [25,35]:
        for step_size in [10, ]:
            for budget in [15]:
                for threshold in [0.7]:
                    for recur_depth in [25]:
                        if recur_depth > start: 
                            continue
                        args.start_step = start
                        args.step_size = step_size
                        args.budget = budget
                        args.threshold = threshold
                        args.recur_depth = recur_depth
                        processed_args = deepcopy(args)
                        processed_args = get_config(args=processed_args)
                        stable_diffusion_sampler = StableDiffusionSampler(processed_args)
                        bfs_guidance = DFSGuidance(processed_args)
                        pipeline = T2IPipeline(processed_args, stable_diffusion_sampler, bfs_guidance)
                        score, results = pipeline.sample()
                        compute = results['compute']
                        with open(csv_path, 'a') as csv_file:
                            csv_file.write(f"{compute},{recur_depth}, {start},{step_size}, {budget}, {threshold}, {score}\n")
    

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample with DFS guidance")
    parser.add_argument('--dataset', type=str, default='color', help='Dataset to sample from')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for sampling')
    cli_args = parser.parse_args()

    sample(cli_args.dataset, cli_args.device)