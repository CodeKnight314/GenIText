from ..models import *
from typing import List, Dict, Union
from tqdm import tqdm
from glob import glob
import os
from PIL import Image

class End2EndCaptionPipeline():
    def __init__(self, model: str, config: str):
        """Initialize end-to-end captioning pipeline.
        
        Args:
            model: Model to use for captioning
        """
        self.models = {
            "llava": [LlavaModel, LlavaProcessor],
            "vit_gpt2": [ViTGPT2Model, VITGPT2Processor], 
            "blipv2_style": [BLIPv2_StyleID, BLIPv2_Processor]
        }
        if model not in self.models:
            raise ValueError(f"[ERROR] Model '{model}' not found.")
        else: 
            self.model = self.models[model][0](config)
            self.processor = self.models[model][1](config)
        
        self.batch_size = self.processor.batch_size
        self.img_h = self.processor.img_h if hasattr(self.processor, "img_h") else None
        self.img_w = self.processor.img_w if hasattr(self.processor, "img_w") else None
         
    def generate_captions(self, inputs: Union[List[str], str]) -> List[Dict[str, str]]:
        """
        Generate captions for a list of images.
        
        Args:
            inputs: List of image paths
        
        Returns:
            List of dictionaries formatted as {"image": str, "caption": str}
        """
        if isinstance(inputs, str):
            inputs = glob(os.path.join(inputs, "*"))

        inputs = [Image.open(img) for img in inputs if os.path.isfile(img)]
        if self.img_h and self.img_w:
            inputs = [img.resize((self.img_w, self.img_h)) for img in inputs]
        
        caption_results = []
        
        for img_batch_idx in tqdm(range(0, len(inputs), self.batch_size)):
            img_batch = inputs[img_batch_idx:img_batch_idx + self.batch_size]
            preprocessed_imgs = self.processor.preprocess(img_batch)
            outputs = self.model.caption_images(preprocessed_imgs)
            captions = self.processor.postprocess(outputs)
            
            for i, img in enumerate(img_batch):
                row = {"image": img, "caption": captions[i]}    
                caption_results.append(row)    
                                
        return caption_results