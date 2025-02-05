import os
from tqdm import tqdm
import yaml
from typing import Dict, List
            
def save_images(captions: List[Dict[str, str]], output_path: str = "output/samples/"):
    """
    Save a list of images to a directory.
    
    Args:
        captions (List[Dict[str, str]]): List of images to save.
        output_path (str): Path to save the images.
    """
    os.makedirs(output_path, exist_ok=True)
    for i, img in enumerate(tqdm(captions, total=len(captions), desc=f"Saving images to {output_path}")):
        with open(os.path.join(output_path, "captions", f"result_{i}.txt"), 'w') as f:
            f.write(img["caption"])
        img.save(os.path.join(output_path, "images", f"result_{i}.png"))
    print("[INFO] Finished saving images")
    
def save_captions(captions: List[Dict[str, str]], output_path: str = "output"):
    """
    Save a list of dictionaries to a csv file.
    
    Args:
        captions (List[Dict[str, str]]): List of dictionaries to save.
        output_path (str): Path to save the csv file.
    """
    os.makedirs(output_path, exist_ok=True)
    file_name = os.path.join(output_path, "captions.csv")
    
    with open(file_name, 'w') as f: 
        f.write(','.join(captions[0].keys()) + '\n')
        f.write('\n')
        for row in captions: 
            f.write(','.join(str(x) for x in row.values()) + '\n')
    print(f"Captions saved to {file_name}")

def load_config(config_path: str):
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