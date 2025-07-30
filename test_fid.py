from evaluations.image import ImageEvaluator
from utils.configs import Arguments
import numpy as np
from PIL import Image

args = Arguments()
args.guide_networks = ['google/vit-base-patch16-224']
evaluator = ImageEvaluator(args)
npy = np.load("logs/guidance_name=bfs+recur_steps=1+iter_steps=4/model=models_openai_imagenet.pt/guide_net=google_vit-base-patch16-224/bon_guidance=google_vit-base-patch16-224/target=222/rho=0.2-increase+mu=0.4-increase+sigma=0.1-decrease/start=25+step_size=25/particles=8+temp=0.0/images.npy")
images = [Image.fromarray(img) for img in npy]
evaluator._compute_validity(images, [222], guide_networks=args.guide_networks)
ood = evaluator._get_ood(images, batchsize=256, guide_network=args.guide_networks[0])
print(f"ood: {ood}")