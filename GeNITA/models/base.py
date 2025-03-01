from abc import ABC, abstractmethod
from typing import Dict, List, Union
from PIL import Image
import torch
import yaml
import os

class BaseModel(ABC): 
    def __init__(self, config: str):
        config = self.load_config(config)

        if config["model"]["device"] == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else: 
            self.device = config["model"]["device"]
        
        self.model_id = config["model"]["model_id"]
        self.model_config = config["model"]
        self.gen_config = config["generation"]
        self.auto_batch = config["model"]["auto_batch"]
    
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
        
    def load_config(self, config_path: str):
        """
        Load configuration from a yaml file.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        if(not os.path.exists(config_path)):
            raise ValueError(f"[ERROR] Config file {config_path} not found.")
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"[ERROR] Error loading config file {config_path}: {e}")
    
class BaseProcessor(ABC): 
    def __init__(self, config: str):
        config = self.load_config(config)
        
        if config["model"]["device"] == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else: 
            self.device = config["model"]["device"]
        
        self.model_id = config["model"]["model_id"]
        self.processor_config = config["processor"]
        self.img_h = self.processor_config["img_h"]
        self.img_w = self.processor_config["img_w"]
        self.batch_size = self.processor_config["batch_size"]
        
    @abstractmethod
    def load_processor(self) -> None:  
        pass
    
    @abstractmethod
    def preprocess(self, images: Union[List[Image.Image], Image.Image]) -> torch.Tensor:
        pass
    
    @abstractmethod
    def postprocess(self, outputs: torch.Tensor) -> Union[str, List[str]]:
        pass
    
    def load_config(self, config_path: str):
        """
        Load configuration from a yaml file.
        
        Args:
            config_path (str): Path to the configuration file.
        """
        if(not os.path.exists(config_path)):
            raise ValueError(f"[ERROR] Config file {config_path} not found.")
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise Exception(f"[ERROR] Error loading config file {config_path}: {e}")
        
    @abstractmethod
    def clean_string(self, text: str):
        pass
    
        