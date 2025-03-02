from GenIText.models.base import BaseModel, BaseProcessor
import torch 
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import List, Union, Dict, Optional
from PIL import Image

class LlavaModel(BaseModel):
    def __init__(self, config: Optional[str]):
        super().__init__(config)
        self.load_model()
    
    def load_model(self) -> None:
        """
        Loads Model based on the configuration when initializing the class.
        """
        if self.model_config["quantize"]["enabled"]:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_8bit=(self.model_config["quantize"]["quant_type"] == "8bit"),
                load_in_4bit=(self.model_config["quantize"]["quant_type"] == "4bit"),
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=["lm_head"]
            )
            
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id, 
                quantization_config=quant_config,
                low_cpu_mem_usage=self.model_config["low_cpu_mem"],
                device_map="auto"
            )
        else:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id, 
                low_cpu_mem_usage=self.model_config["low_cpu_mem"]
            ).to(self.device)
        
        self.model.config.eos_token_id = self.model.config.pad_token_id = 2
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.pad_token_id = 2
            self.model.generation_config.eos_token_id = 2
    
    def caption_images(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Captions a given image based on the inputs provided.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Inputs to the model.

        Returns:
            torch.Tensor: Generated Caption.
        """
        with torch.no_grad(): 
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

class LlavaProcessor(BaseProcessor):
    def __init__(self, config: Optional[str] = None):
        super().__init__(config)
        self.load_processor()
        self.default_prompt = self.processor_config["default_prompt"]

    def load_processor(self):
        """
        Loads Processor based on the configuration when initializing the class.
        """
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, 
            use_fast=True
        )

    def preprocess(self, images: Union[List[Image.Image], Image.Image], prompts : Union[List[str], str] = None) -> torch.Tensor:
        """
        Preprocesses the inputs for the model.
        
        Args:
            images (Union[List[Image.Image], Image.Image]): Images to be processed.
            prompts (Union[List[str], str]): Prompts to be processed.
            
        Returns:
            torch.Tensor: Processed Inputs.
        """
        if isinstance(images, Image.Image):
            images = [images]
        
        for i, img in enumerate(images):
            if img.size != (self.img_w, self.img_h):
                images[i] = img.resize((self.img_w, self.img_h))

        if isinstance(prompts, list):
            processed_prompts = []
            for prompt in prompts:
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"}
                    ],
                }]
                processed_prompts.append(self.processor.apply_chat_template(conversation, add_generation_prompt=True))
            prompts = processed_prompts
        else:
            if prompts is None:
                prompts = self.default_prompt
                
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompts},
                    {"type": "image"}
                ],
            }]
            prompts = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)
        elif len(prompts) == 1 and len(images) > 1:
            prompts = prompts * len(images)
        elif len(prompts) != len(images):
            raise ValueError("The number of prompts must match the number of images or be a single prompt.")

        inputs = self.processor(
            text=prompts, 
            images=images, 
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        return inputs
    
    def postprocess(self, outputs: torch.Tensor) -> Union[str, List[str]]:
        """
        Postprocesses the outputs from the model.
        
        Args:
            outputs (torch.Tensor): Outputs from the model.
        
        Returns:
            Union[str, List[str]]: Postprocessed Outputs.
        """
        return self.processor.batch_decode(outputs, skip_special_tokens=True)
    
    def clean_string(self, text: str):
        if "ASSISTANT:" in text:
            text = text[text.find("ASSISTANT:") + len("ASSISTANT:"):].strip()
        
        text = text.strip()
        text = ' '.join(line.strip() for line in text.split("\n"))
        text = ' '.join(text.split())
        
        if text and not text[-1] in ["?", ".", "!"]:
            text += "."
        
        return text
        