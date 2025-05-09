import torch.utils
from transformers import AutoModel, AutoTokenizer
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import numpy as np
import torch
from torchvision import transforms


class OCRGuider:
    def __init__(self,args):
        device = args.device
        letters = args.target
        self.tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=self.tokenizer.eos_token_id,torch_dtype=torch.float16)
        self.model = self.model.eval().to(device)
        labels = self.tokenizer(letters)['input_ids'] + [151645]
        self.labels = torch.tensor(labels).to(device)
        self.letters = letters
        self.context_len = len(labels)

    def get_logits(self,image):
        logits = self.model.chat(self.tokenizer, image, ocr_type='ocr',append_text=self.letters)
        return logits

    def get_guidance(self,x_need_grad,func=lambda x: x, post_process=lambda x: x,return_logp=False,**kwargs):
        x = func(x_need_grad)
        x = post_process(x)
        x = x.half()
        logits = self.get_logits(x)
        logits = logits[...,-self.context_len-1:-1,:]
        logits = logits.view(-1,self.model.config.vocab_size)
        labels = self.labels.view(-1)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        if return_logp:
            return loss
        grad = torch.autograd.grad(loss, x_need_grad)[0]
        return grad


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
    # model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
    # model = model.eval().to('cuda')
    
    # # # input your test image
    # image_file = 'sample.png'

    # # plain texts OCR
    # res = model.chat(tokenizer, image_file, ocr_type='ocr')

    # # format texts OCR:
    # # res = model.chat(tokenizer, image_file, ocr_type='format')
    # print(res)
    # fine-grained OCR:
    # res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_box='')
    # res = model.chat(tokenizer, image_file, ocr_type='format', ocr_box='')
    # res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_color='white')
    # res = model.chat(tokenizer, image_file, ocr_type='format', ocr_color='')

    # multi-crop OCR:
    # res = model.chat_crop(tokenizer, image_file, ocr_type='ocr')
    # res = model.chat_crop(tokenizer, image_file, ocr_type='format')

    # render the formatted OCR results:
    # res = model.chat(tokenizer, image_file, ocr_type='format', render=True, save_render_file = './demo.html')

    # print(res)
    # import PIL
    # image = PIL.Image.open('sample.png').convert('RGB')
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # ])

    # image = transform(image).unsqueeze(0).to('cuda').requires_grad_(True)
    # guider = OCRGuider(letters="laptop",device="cuda")
    # grad = guider.get_guidance(image)
    # labels = tokenizer("coldfoodstorage")["input_ids"] + [151645]
    # print(labels)

    


