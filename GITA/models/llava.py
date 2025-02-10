from .base import BaseModel, BaseProcessor
import torch 
from transformers import LlavaNextProcessor, LlavaForConditionalGeneration
from typing import List, Union, Dict, Optional
from PIL import Image

class LlavaModel(BaseModel):
    def __init__(self, 
                 config: Optional[str]):
        """Initialize LLaVA model with configuration.
        
        Args:
            model_id: HuggingFace model identifier
            config: Path to configuration file
        """
        super().__init__(config)
        self.load_model()
    
    def load_model(self) -> None:
        """Load and configure the LLaVA model."""
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id
        ).to(self.device)
        
        self.model.config.eos_token_id = self.model.config.pad_token_id = 2
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.pad_token_id = 2
            self.model.generation_config.eos_token_id = 2
    
    def caption_images(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate captions for preprocessed image inputs.
        
        Args:
            inputs: Dictionary containing model inputs
            
        Returns:
            Tensor containing generated token sequences
        """
        with torch.no_grad(): 
            outputs = self.model.generate(
                **inputs, 
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

class LlavaProcessor(BaseProcessor):
    def __init__(self, 
                 config: Optional[str] = None):
        """Initialize LLaVA processor with configuration.
        
        Args:
            model_id: HuggingFace model identifier
            config: Path to configuration file
        """
        config = self.load_config(config)
        self.model_id = config["model"]["model_id"]
        if config["model"]["device"] == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.batch_size = config["batch_size"]
        
        self.processor_config = config["processor"]
        self.load_processor()
    
    def load_processor(self):
        self.processor = LlavaNextProcessor.from_pretrained(
            self.model_id
        )
        
        self.processor.image_processor.size = {
            "height": self.processor_config.get("image_size", {}).get("height", 336),
            "width": self.processor_config.get("image_size", {}).get("width", 336)
        }
        self.processor.tokenizer.padding_side = self.processor_config.get("padding_side", "left")

        
    def preprocess(self, images: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
        """Preprocess images and text for model input.
        
        Args:
            images: Single image or list of images
            prompt: Optional custom prompt
            
        Returns:
            Dictionary containing preprocessed inputs
        """
        if prompt is None: 
            prompt = self.processor_config.get("default_prompt", "Describe this image.")
            
        if isinstance(images, Image.Image):
            images = [images]
            
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"}
                ],
            }
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        if len(images) > 1: 
            prompt = [prompt] * len(images)
        
        inputs = self.processor(
            text=prompt, 
            images=images, 
            return_tensors="pt"
        ).to(self.device, dtype=torch.float16)
        
        return inputs
    
    def postprocess(self, outputs: torch.Tensor) -> Union[str, List[str]]:
        """Convert model outputs to human-readable captions.
        
        Args:
            outputs: Model output tensors
            batch_size: Number of images processed
            
        Returns:
            Single caption string or list of captions
        """
        if self.batch_size > 1:
            captions = self.processor.batch_decode(outputs, skip_special_tokens=True)
        else: 
            captions = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return captions