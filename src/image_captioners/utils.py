import os
import importlib
from tqdm import tqdm
import yaml
            
def save_images(images: list, save_pth: str, prompts: list = None):
    """
    Save a list of images to a directory.
    
    Args:
        images (list): List of PIL images to save.
        save_pth (str): Path to save the images.
    """
    os.makedirs(save_pth, exist_ok=True)
    for i, img in enumerate(tqdm(images, total=len(images), desc=f"Saving images to {save_pth}")):
        if prompts is not None: 
            with open(os.path.join(save_pth, f"result_{i}.txt"), 'w') as f:
                f.write(prompts[i])
        img.save(os.path.join(save_pth, f"result_{i}.png"))        
    print("Finished saving images")

def load_config(config_path: str):
    """
    Load configuration from a yaml file.
    
    Args:
        config_path (str): Path to the configuration file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)