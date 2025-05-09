from t2i_pipeline import T2IPipeline
from TFG_args import args, get_config, dataset_to_task
from methods.bfs import BFSGuidance
from methods.base import BaseGuidance
from diffusion.stable_diffusion import StableDiffusionSampler
import os
from copy import deepcopy

def sample(dataset, device):
    args.guidance_name = 'bon'
    # args.model_name_or_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    args.model_name_or_path = 'segmind/SSD-1B'
    args.dataset = dataset
    args.task = dataset_to_task(dataset)
    args.device = device

    csv_path = f"results_{args.dataset}_SSD_bon.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as csv_file:
            csv_file.write("particles, score\n")
    
    for particles in [1,3,5,6,7]:
        args.per_sample_batch_size = particles
        args.temp = 0
        args.resample_interval = 100
        args.num_samples = 1
        processed_args = deepcopy(args)
        processed_args = get_config(args=processed_args)
        stable_diffusion_sampler = StableDiffusionSampler(processed_args)
        bfs_guidance = BFSGuidance(processed_args)
        pipeline = T2IPipeline(processed_args, stable_diffusion_sampler, bfs_guidance)
        score, results = pipeline.sample(1)
        compute = results['compute']
        with open(csv_path, 'a') as csv_file:
            csv_file.write(f"{particles}, {score}\n")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample with BFS guidance")
    parser.add_argument('--dataset', type=str, default='color', help='Dataset to sample from')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for sampling')
    cli_args = parser.parse_args()

    sample(cli_args.dataset, cli_args.device)