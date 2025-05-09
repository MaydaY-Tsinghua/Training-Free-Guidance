import argparse
import os

import torch

from tqdm import tqdm, trange
from typing import List

import json
from tqdm.auto import tqdm
import sys
import re
import spacy
nlp = spacy.load("en_core_web_sm")

# current_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, current_dir)
# from models.blip_vqa import blip_vqa
# print(sys.path)
from BLIPvqa_eval.models.blip_vqa import blip_vqa
# sys.path.pop(0)

import yaml
config = '/datapool/data2/home/linhw/zhangxiangcheng/DiffTTS/T2I/TFG/T2I-CompBench/BLIPvqa_eval/configs/vqa.yaml' #todo config file
config = yaml.load(open(config, 'r'), Loader=yaml.Loader)
config['inference'] = 'vqa_prob'

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question

def create_annotation_for_BLIP(prompt: str, images:torch.Tensor,  np_num=8):
    f = prompt
    doc = nlp(f)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        if chunk.text not in ['top', 'the side', 'the left', 'the right']:  # todo remove some phrases
            noun_phrases.append(chunk.text)

    annotations = []
    # cnt=0
    for np_index in range(np_num):
        #output annotation.json
        image_dict = {}
        image_dict['image'] = images
        image_dict['question_id']= np_index
        
        if(len(noun_phrases)>np_index):
            q_tmp = noun_phrases[np_index]
            image_dict['question']=f'{q_tmp}?'
        else:
            break
            image_dict['question'] = ''
        image_dict['dataset']="color"
        image_dict['question']=[pre_question(image_dict['question'])] * images.shape[0]
        # cnt+=1
        annotations.append(image_dict)
        # print('Number of Processed Images:', len(annotations))
    return annotations


from utils.configs import Arguments
from torchvision import transforms

class BLIPGuider:
    def __init__(self, args: Arguments):
        self.args = args
        self.model = blip_vqa(pretrained=config['pretrained'], image_size=config['image_size'],
                       vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
        self.model = self.model.to(args.device)
        self.model.half()
        self.model.eval()
        self.transform = transforms.Compose([transforms.Resize((config['image_size'],config['image_size']),interpolation=transforms.InterpolationMode.BICUBIC),                
                                            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                                            ])
        self.device = torch.device(args.device)
    
    def set_prompt(self, prompt: str):
        self.prompt = prompt
    
    def get_guidance(self, x, func=lambda x: x, post_process=lambda x: x, return_logp=True, **kwargs):
        assert return_logp, "BLIP can not take grad"
        x = func(x)
        x = post_process(x)
        x = self.transform(x)

        annotations = create_annotation_for_BLIP(self.prompt, x)
        results = []
        for image_dict in annotations:
            image = image_dict['image']
            question = image_dict['question']
            question_id = image_dict['question_id']
            with torch.no_grad():
                probs = self.model(image, question, train=False, inference="vqa_prob", output_type='pt')
                logprobs = torch.log(probs)
                results.append(logprobs)
        results = torch.stack(results, dim=0)
        score = results.sum(dim=0)
        return score


        
    