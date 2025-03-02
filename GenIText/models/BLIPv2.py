from PIL import Image 
from transformers import Blip2ForConditionalGeneration, Blip2Processor
import torch 
from typing import List, Union
from GenIText.models.base import BaseModel, BaseProcessor

class BLIPv2Model(BaseModel): 
    def __init__(self, config: str): 
        super().__init__(config)
        self.load_model()
        
    def load_model(self):
        if self.model_config["quantize"]["enabled"]: 
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_8bit=(self.model_config["quantize"]["quant_type"] == "8bit"),
                load_in_4bit=(self.model_config["quantize"]["quant_type"] == "4bit"),
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=["lm_head"]
            )
            
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id,
                quantization_config=quant_config, 
                low_cpu_mem_usage=self.model_config["low_cpu_mem"],
                device_map="auto"
            )

        else: 
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_id, 
                low_cpu_mem_usage=self.model_config["low_cpu_mem"]
            ).to(self.device)
    
    def caption_images(self, inputs: dict): 
        with torch.no_grad(): 
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens = self.gen_config["max_new_tokens"],
                min_new_tokens = self.gen_config["min_new_tokens"],
                num_beams = self.gen_config["num_beams"],
                do_sample = self.gen_config["do_sample"],
                temperature = self.gen_config["temperature"],
                top_k = self.gen_config["top_k"],
                top_p = self.gen_config["top_p"],
                repetition_penalty = self.gen_config["repetition_penalty"],
                length_penalty = self.gen_config["length_penalty"],
                no_repeat_ngram_size = self.gen_config["no_repeat_ngram_size"],
                early_stopping = self.gen_config["early_stopping"],
                return_dict_in_generate = self.gen_config["return_dict_in_generate"],
                output_scores = self.gen_config["output_scores"],
            )
            
        return outputs 
    
class BLIPv2_Processor(BaseProcessor):
    def __init__(self, config: str): 
        super().__init__(config)
        self.load_processor()
        self.default_prompt = self.processor_config["default_prompt"]
        
    def load_processor(self):
        self.processor = Blip2Processor.from_pretrained(
            self.model_id
        )
        
    def preprocess(self, images: Union[List[Image.Image], Image.Image], prompts: Union[List[str], str] = None): 
        if(isinstance(images, Image.Image)): 
            images = [images]
        
        for i, img in enumerate(images):
            if img.size != (self.img_w, self.img_h):
                images[i] = img.resize((self.img_w, self.img_h))
        
        if prompts is None: 
            prompts = [self.default_prompt] * len(images)
        elif(isinstance(prompts, str)):
            prompts = [prompts] * len(images)
            
        if len(prompts) != len(images): 
            raise ValueError("Number of prompts must match the number of images.")
        
        return self.processor(images=images, text=prompts, return_tensors="pt").to(self.device)
    
    def postprocess(self, outputs: List[torch.Tensor]): 
        if isinstance(outputs, dict) and "sequences" in outputs:
            outputs = outputs["sequences"]
        captions = self.processor.batch_decode(outputs, skip_special_tokens=True)            
        captions = [caption.replace(self.default_prompt, "") for caption in captions]
        return captions

    def clean_string(self, text: str):
        return text