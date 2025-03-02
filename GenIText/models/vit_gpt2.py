from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch 
from GenIText.models.base import BaseModel, BaseProcessor
from typing import List, Union
from PIL import Image

class ViTGPT2Model(BaseModel):
    def __init__(self, config: str): 
        super().__init__(config)
        self.load_model()
        
    def load_model(self):
        """
        Load the model from the model_id and model_config
        """
        if self.model_config["quantize"]["enabled"]:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_8bit=(self.model_config["quantize"]["quant_type"] == "8bit"),
                load_in_4bit=(self.model_config["quantize"]["quant_type"] == "4bit"),
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=["lm_head"]
            )
            
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_id, 
                quantization_config=quant_config,
                low_cpu_mem_usage=self.model_config["low_cpu_mem"],
            )
        else:
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.model_id, 
                low_cpu_mem_usage=self.model_config["low_cpu_mem"]
            ).to(self.device)
        
    def caption_images(self, inputs: torch.Tensor):
        """
        Generate captions for the input images.
        
        Args:
            inputs (torch.Tensor): Input images to generate captions for.
            
        Returns:
            List of generated captions for the input images.
        """
        with torch.no_grad(): 
            outputs = self.model.generate(
                inputs,
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
        
class VITGPT2Processor(BaseProcessor): 
    def __init__(self, config: str): 
        super().__init__(config)
        self.load_processor()
        
        self.default_prompt = self.processor_config["default_prompt"]
        
    def load_processor(self):
        """
        Load the processor from the model_id
        """
        self.feature_extractor = ViTImageProcessor.from_pretrained(
            self.model_id
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            
        )
        
    def preprocess(self, images: Union[List[Image.Image], Image.Image], prompts:Union[List[str], str] = None): 
        """
        Preprocess the input images.
        """
        if prompts is None: 
            prompts = self.default_prompt
        elif isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(images, Image.Image): 
            images = [images]
        
        for i, img in enumerate(images):
            if img.size != (self.img_w, self.img_h):
                images[i] = img.resize((self.img_w, self.img_h))
            
        pixel_values = self.feature_extractor(images=images, return_tensors="pt")["pixel_values"]
        return pixel_values.to(self.device)
    
    def postprocess(self, outputs: torch.Tensor):
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds
    
    def clean_string(self, text: str):
        return text