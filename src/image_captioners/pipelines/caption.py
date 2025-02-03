from models import *
from typing import List, Union, Dict, Optional
from tqdm import tqdm

class End2EndCaptionPipe():
    def __init__(self, model: str, config: str):
        """Initialize end-to-end captioning pipeline.
        
        Args:
            model: Model to use for captioning
        """
        self.models = {
            "llava": [LlavaModel, LlavaProcessor]
        }
        if model not in self.models:
            raise ValueError(f"Model '{model}' not found.")
        else: 
            self.model = self.models[model][0](config)
            self.processor = self.models[model][1](config)
        
        self.batch_size = config["batch_size"]
         
    def generate_captions(self, inputs: List[str]):
        captions = {}
        
        for img_batch_idx in tqdm(range(0, len(inputs), self.batch_size)):
            img_batch = inputs[img_batch_idx:img_batch_idx + self.batch_size]
            preprocessed_imgs = self.processor.preprocess(img_batch)
            outputs = self.model.generate(preprocessed_imgs)
            captions = self.processor.postprocess(outputs)
            
            for i, img in enumerate(img_batch):
                captions[img] = captions[i]
                
        return captions

            
        
            
            
            