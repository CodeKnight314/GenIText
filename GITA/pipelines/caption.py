from ..models import *
from typing import List, Dict
from tqdm import tqdm

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
        
        self.batch_size = config["batch_size"]
         
    def generate_captions(self, inputs: List[str]) -> List[Dict[str, str]]:
        """
        Generate captions for a list of images.
        
        Args:
            inputs: List of image paths
        
        Returns:
            List of dictionaries formatted as {"image": str, "caption": str}
        """
        caption_results = []
        
        for img_batch_idx in tqdm(range(0, len(inputs), self.batch_size)):
            img_batch = inputs[img_batch_idx:img_batch_idx + self.batch_size]
            preprocessed_imgs = self.processor.preprocess(img_batch)
            outputs = self.model.generate(preprocessed_imgs)
            captions = self.processor.postprocess(outputs)
            
            for i, img in enumerate(img_batch):
                row = {"image": img, "caption": captions[i]}    
                caption_results.append(row)    
                                
        return caption_results

            
        
            
            
            