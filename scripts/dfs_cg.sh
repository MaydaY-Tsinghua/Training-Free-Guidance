CUDA_VISIBLE_DEVICES=7
data_type=image
image_size=256
dataset="imagenet"
model_name_or_path='models/openai_imagenet.pt'

task=label_guidance_time
guide_network='ckpts/models/timeclassifier_imagenet.pt'
bon_guidance='google/vit-base-patch16-224'
target=222

train_steps=1000
inference_steps=100
eta=1.0
clip_x0=True
seed=42
logging_dir='logs'
per_sample_batch_size=1
num_samples=256
logging_resolution=512
guidance_name='dfs-cg'
eval_batch_size=32
wandb=False


rho=2
mu=0.5
sigma=0.1
eps_bsz=1
iter_steps=4
guidance_strength=10

start=70
step_size=25
threshold=0.6
budget=4
recur_depth=70

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py \
    --data_type $data_type \
    --task $task \
    --image_size $image_size \
    --bon_guidance $bon_guidance \
    --guidance_strength $guidance_strength \
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
    --budget $budget \
    --threshold $threshold \
    --recur_depth $recur_depth \
    --start $start \
    --step_size $step_size 


