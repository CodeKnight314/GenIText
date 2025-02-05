from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch 
from base import BaseModel, BaseProcessor 
from typing import List, Union
from utils import load_config
from PIL import Image

class ViTGPT2Model(BaseModel):
    def __init__(self, config: str): 
        config = load_config(config)
        
        self.model_id = config["model"]["model_id"]
        if config["model"]["device"] == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.gen_config = config["generation"]
        
    def load_model(self):
        self.model = VisionEncoderDecoderModel.from_pretrained(
            self.model_id, 
        )
        
    def caption_images(self, inputs: torch.Tensor):
        with torch.no_grad(): 
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.gen_config.get("max_new_tokens", 100),
                num_beams=self.gen_config.get("num_beams", 5),
                do_sample=self.gen_config.get("do_sample", True),
                temperature=self.gen_config.get("temperature", 0.7),
                top_p=self.gen_config.get("top_p", 0.95),
                repetition_penalty=self.gen_config.get("repetition_penalty", 1.5),
                min_new_tokens=self.gen_config.get("min_new_tokens", 1),
                early_stopping=self.gen_config.get("early_stopping", True),
                length_penalty=self.gen_config.get("length_penalty", 1.0),
                no_repeat_ngram_size=self.gen_config.get("no_repeat_ngram_size", 3),
            )
            
        return outputs
        
    def model_info(self):
        return {
            "model_id": self.model_id,
            "device": self.device, 
            "config": self.gen_config
        }
        
class VITGPT2Processor(BaseProcessor): 
    def __init__(self, config: str): 
        config = load_config(config)
        self.model_id = config["model"]["model_id"]
        if config["model"]["device"] == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.batch_size = config["batch_size"]
        
        self.processor_config = config["processor"]
        self.load_processor()
        
    def load_processor(self):
        self.feature_extractor = ViTImageProcessor.from_pretrained(
            self.model_id
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id
        )
        
    def preprocess(self, images: Union[List[Image.Image], Image.Image]): 
        if isinstance(images, Image.Image): 
            images = [images]
        pixel_values = self.feature_extractor(images, return_tensors="pt")
        return pixel_values.to(self.device)
    
    def postprocess(self, outputs: torch.Tensor):
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds
        
            
        
        
        