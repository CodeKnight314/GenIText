from PIL import Image 
from transformers import Blip2ForConditionalGeneration, Blip2Processor 
import torch 
from utils import load_config
from typing import List, Union, Dict

class BLIPv2_StyleID(): 
    def __init__(self, config: str): 
        config = load_config(config)
        self.model_id = config["model"]["model_id"]
        if(config["model"]["device"] == "cuda" and torch.cuda.is_available()): 
            self.device = "cuda"
        else: 
            self.device = "cpu"
        
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id, 
            load_in_8bit=config["model"].get("load_in_8bit", True), 
            torch_dtype=torch.float16 if config["model"].get("use_fp16", False) else torch.float32
        )
        
        self.processor = Blip2Processor.from_pretrained(
            self.model_id
        )
        
        self.style_prompt = config["model"].get("style_prompt", "What is the artistic or visual style of this image? Describe in 3-4 words.")
        self.gen_config = config["generation"]
        
    def preprocess(self, images: Union[List[Image.Image], Image.Image]): 
        if(isinstance(images, Image.Image)): 
            images = [images]
        return self.processor(images=images, return_tensors="pt").to(self.device)
    
    def generate_captions(self, inputs: torch.Tensor): 
        with torch.no_grad(): 
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=self.gen_config.get("max_new_tokens", 20),
                num_beams = self.gen_config.get("num_beams", 3),
                temperature = self.gen_config.get("temperature", 0.3), 
                do_sample = self.gen_config.get("do_sample", True),
                reptition_penalty = self.gen_config.get("repetition_penalty", 1.2),
            )
            
        return outputs 
    
    def postprocess(self, outputs: List[torch.Tensor]): 
        captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return captions