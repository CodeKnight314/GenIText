from PIL import Image 
from transformers import Blip2ForConditionalGeneration, Blip2Processor 
import torch 
from typing import List, Union
from base import BaseModel, BaseProcessor

class BLIPv2_StyleID(BaseModel): 
    def __init__(self, config: str): 
        super().__init__(config)
        self.load_model()
        
    def load_model(self):
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id
        )
    
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
    
class BLIPv2_Processor(BaseProcessor):
    def __init__(self, config: str): 
        super().__init__(config)
            
        self.style_prompt = config["model"].get("style_prompt", "What is the artistic or visual style of this image? Describe in 3-4 words.")
        self.load_processor()
        
    def load_processor(self):
        self.processor = Blip2Processor.from_pretrained(
            self.model_id
        )
        
    def preprocess(self, images: Union[List[Image.Image], Image.Image]): 
        if(isinstance(images, Image.Image)): 
            images = [images]
        return self.processor(images=images, return_tensors="pt").to(self.device)
    
    def postprocess(self, outputs: List[torch.Tensor]): 
        captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return captions
