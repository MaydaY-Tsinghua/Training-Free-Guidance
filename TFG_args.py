from utils.configs import Arguments
from utils.utils import get_config

args = Arguments(
    data_type="text2image",
    image_size=1024,
    dataset="color",
    model_name_or_path='segmind/SSD-1B',
    # model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",

    task='blip_vqa',
    guide_network='openai/clip-vit-base-patch16',
    # target='./data/wikiart/2.png',
    target="peace",

    train_steps=1000,
    inference_steps=50,
    eta=0.0,
    clip_x0=False,
    seed=2,
    logging_dir='logs',
    per_sample_batch_size=1,
    num_samples=1,
    logging_resolution=512,
    guidance_name='tfg',
    eval_batch_size=1,
    wandb=False,

    rho=0,
    mu=1,
    sigma=0,
    eps_bsz=1,

    recur_steps=1,
    iter_steps=1,


)
# args = get_config(args=args)


def dataset_to_task(dataset):
    if dataset == 'color' or dataset == 'shape' or dataset == 'texture':
        return 'blip_vqa'
    elif dataset == 'numeracy' or dataset == 'spatial':
        return 'unidet'
    elif dataset == 'non-spatial':
        return 'clip_score'
   