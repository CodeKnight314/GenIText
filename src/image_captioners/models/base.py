from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union
from PIL import Image
import torch
from utils import load_config

class BaseModel(ABC): 
    def __init__(self, config: str):
        config = load_config(config)
        self.model_id = config["model"]["model_id"]
        if config["model"]["device"] == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.gen_config = config["generation"]
    
    @abstractmethod
    def load_model(self) -> None:
        pass
    
    @abstractmethod
    def caption_images(self, images: Union[List[Image.Image], Image.Image], **kwargs) -> List[str]:
        pass 
    
    def model_info(self) -> Dict[str, Union[str, dict]]:
        """Return model information and configuration.
        
        Returns:
            Dictionary containing model metadata and settings
        """
        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "config": self.gen_config
        }
    
class BaseProcessor(ABC): 
    def __init__(self, config: str):
        config = load_config(config)
        self.model_id = config["model"]["model_id"]
        if config["model"]["device"] == "cuda" and torch.cuda.is_available(): 
            self.device = "cuda"
        else: 
            self.device = "cpu"
    
    @abstractmethod
    def load_processor(self) -> None:  
        pass
    
    @abstractmethod
    def preprocess(self, images: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
        pass
    
    @abstractmethod
    def postprocess(self, outputs: torch.Tensor) -> Union[str, List[str]]:
        pass
    
        