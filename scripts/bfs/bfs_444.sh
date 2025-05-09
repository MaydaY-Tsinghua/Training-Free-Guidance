#!/bin/bash
#SBATCH --job-name=tfg_222              # Job name
#SBATCH -o status/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e status/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1                    # Request 1 GPU
#SBATCH --mem=20G                       # Memory
#SBATCH --time=6:00:00                 # Max runtime
#SBATCH --partition=gpu                 # Adjust to your cluster
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=504985967@qq.com
# Load modules or set up env if needed
# module load python/3.8 cuda/11.7  (example)

# Set CUDA device (optional, Slurm will often handle this)
# export CUDA_VISIBLE_DEVICES=0

# Parameters
data_type=image
image_size=256
dataset="imagenet"
model_name_or_path='models/openai_imagenet.pt'

task=label_guidance
guide_network='google/vit-base-patch16-224'
bon_guidance='google/vit-base-patch16-224'
target=444

train_steps=1000
inference_steps=100
eta=1.0
clip_x0=True
seed=42
logging_dir='logs'
per_sample_batch_size=2
num_samples=256
logging_resolution=512
guidance_name='bfs'
bon_rate=1
eval_batch_size=32
wandb=False

rho=0.2
mu=0.8
sigma=0.01
eps_bsz=1
iter_steps=4

start=25
step_size=25
temp=0.0

# Run
python main.py \
    --data_type $data_type \
    --task $task \
    --bon_rate $bon_rate \
    --image_size $image_size \
    --dataset $dataset \
    --guide_network $guide_network \
    --logging_resolution $logging_resolution \
    --model_name_or_path $model_name_or_path \
    --train_steps $train_steps \
    --inference_steps $inference_steps \
    --target $target \
    --iter_steps $iter_steps \
    --eta $eta \
    --clip_x0 $clip_x0 \
    --rho $rho \
    --mu $mu \
    --sigma $sigma \
    --eps_bsz $eps_bsz \
    --wandb $wandb \
    --seed $seed \
    --logging_dir $logging_dir \
    --per_sample_batch_size $per_sample_batch_size \
    --num_samples $num_samples \
    --guidance_name $guidance_name \
    --eval_batch_size $eval_batch_size \
    --temp $temp \
    --bon_guidance $bon_guidance \
    --start $start \
    --step_size $step_size \