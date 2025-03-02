from typing import List, Dict, Union
from tqdm import tqdm
from glob import glob
import os
from PIL import Image
import torch
import numpy as np
import importlib.resources

from GenIText.models.llava import LlavaModel, LlavaProcessor
from GenIText.models.vit_gpt2 import ViTGPT2Model, VITGPT2Processor
from GenIText.models.BLIPv2 import BLIPv2Model, BLIPv2_Processor

class End2EndCaptionPipeline():
    models = {
        "llava": [LlavaModel, LlavaProcessor],
        "vit_gpt2": [ViTGPT2Model, VITGPT2Processor], 
        "blipv2": [BLIPv2Model, BLIPv2_Processor]
    }
    
    def __init__(self, model: str, config: str):
        """Initialize end-to-end captioning pipeline.
        
        Args:
            model: Model to use for captioning
        """
        if model not in End2EndCaptionPipeline.models:
            raise ValueError(f"[ERROR] Model '{model}' not found.")
        else: 
            if config is None: 
                config = self._get_default_config_path(model)
            self.model = End2EndCaptionPipeline.models[model][0](config)
            self.processor = End2EndCaptionPipeline.models[model][1](config)
        
        self.batch_size = self.processor.batch_size if hasattr(self.processor, "batch_size") else 1
        self.img_h = self.processor.img_h if hasattr(self.processor, "img_h") else None
        self.img_w = self.processor.img_w if hasattr(self.processor, "img_w") else None
        
        if self.model.auto_batch:
            batch_result = self.check_gpu_memory_usage()
            if batch_result["success"]:
                self.batch_size = max(self.batch_size, batch_result["recommended_max_batch_size"])
            else:
                raise MemoryError(f"[WARNING] Auto-batch failed. Model is unable to handle images at size ({self.img_h}, {self.img_w}). Model uses ")
            
    def _get_default_config_path(self, model_name: str) -> str:
        """Get the path to the default config file for a model.
        
        This works whether the package is installed via pip or in development mode.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to default config file
        """
        try:
            with importlib.resources.path('GenIText.configs', f'{model_name}_config.yaml') as path:
                return str(path)
        except (ImportError, ModuleNotFoundError):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(base_dir, 'configs', f'{model_name}_config.yaml')
                
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

        inputs = [img for img in inputs if os.path.isfile(img)]
        
        caption_results = []
        
        for img_batch_idx in tqdm(range(0, len(inputs), self.batch_size)):
            img_path_batch = inputs[img_batch_idx:img_batch_idx + self.batch_size]
            img_batch = [Image.open(img).convert("RGB") for img in img_path_batch]
            if self.img_h and self.img_w:
                img_batch = [img.resize((self.img_w, self.img_h)) for img in img_batch]
            preprocessed_imgs = self.processor.preprocess(img_batch)
            outputs = self.model.caption_images(preprocessed_imgs)
            captions = self.processor.postprocess(outputs)
            
            for i, img_path in enumerate(img_path_batch):
                row = {"image": img_path, "caption": captions[i]}    
                caption_results.append(row)    
                                
        return caption_results
    
    def check_gpu_memory_usage(self, safety_margin: float = 0.9):
        """
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated()
        
        img_batch = [Image.fromarray(np.random.randint(0, 255, (self.img_h, self.img_w, 3), dtype=np.uint8)).convert("RGB") for _ in range(self.batch_size)]
        
        try: 
            img_batch = self.processor.preprocess(img_batch)
            
            with torch.no_grad():
                _ = self.model.caption_images(img_batch)
                
            peak_memory = torch.cuda.max_memory_allocated()
            memory_per_batch = peak_memory - initial_mem
            
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            max_batch_size = int((total_memory * safety_margin) / memory_per_batch)
            
            return {
                'success': True,
                'memory_per_image': memory_per_batch / self.batch_size,
                'memory_per_batch': memory_per_batch,
                'peak_memory': peak_memory,
                'total_gpu_memory': total_memory,
                'available_memory': total_memory * safety_margin,
                'recommended_max_batch_size': max_batch_size,
                'current_batch_memory_percentage': (memory_per_batch / total_memory) * 100
            }
            
        except torch.cuda.OutOfMemoryError as e:
            return {
                "success": False,
                "total_gpu_memory": torch.cuda.get_device_properties(0).total_memory,
                "error": str(e)
            }
        
        finally: 
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()