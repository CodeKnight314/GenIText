import os
from tqdm import tqdm
from typing import Dict, List
from random import sample
import requests
import zipfile
from PIL import Image
import json
import shutil

def download_dataset(url: str = None, path: str = "dataset/"):
    if url is None:
        url = "https://www.kaggle.com/api/v1/datasets/download/hadiepratamatulili/anime-vs-cartoon-vs-human"

    os.makedirs(path, exist_ok=True)
    download_path = os.path.join(path, "anime-vs-cartoon-vs-human.zip")

    if not os.path.exists(download_path):
        if os.path.exists(os.path.join(path, "Data")) and len(os.listdir(os.path.join(path, "Data", "anime"))) > 0 and len(os.listdir(os.path.join(path, "Data", "human"))) > 0 and len(os.listdir(os.path.join(path, "Data", "cartoon"))) > 0:
            print(f"[INFO] {os.path.join(path, 'Data')} already exists")
            return os.path.join(path, "Data/")

        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024

            with open(download_path, "wb") as f, tqdm(
                desc="Downloading Dataset from kaggle URL",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(block_size):
                    f.write(chunk)
                    bar.update(len(chunk))

            print(f"[INFO] Successfully downloaded to {download_path}")
        else:
            print(f"Download failed with status code: {response.status_code}")
    else: 
        print(f"[INFO] {download_path} already exists")
        
    print("[INFO] Extracting files...")
    with zipfile.ZipFile(download_path, 'r') as zip_ref, tqdm(
        desc="Extracting files",
        total=len(zip_ref.namelist()),
        unit="files",
    ) as bar:
        for file in zip_ref.namelist():
            zip_ref.extract(file, path)
            bar.update(1)
        print(f"[INFO] Extracted all files to {path}")
    
    os.remove(download_path)
    return os.path.join(path, "Data/")

def cut_data(dataset_path: str, sample_threshold: int):
    """
    Cut the dataset to a certain number of samples.
    
    Args:
        dataset_path (str): Path to the dataset.
        sample_threshold (int): Number of samples to keep.
    """
    if(len(os.listdir(dataset_path)) > sample_threshold):
        files = os.listdir(dataset_path)
        for file in sample(files, len(files) - sample_threshold):
            os.remove(os.path.join(dataset_path, file))
        print(f"[INFO] Removed {len(files) - sample_threshold} files")

def prepare_data(sample_threshold: int = 100, target_dir: str = "dataset/"):
    """
    Prepare the test dataset for training.
    
    Args:
        sample_threshold (int): Number of samples to keep.
        target_dir (str): Path to save the dataset.
        
    Returns:
        Tuple[List[str], List[str], List[str]]: List of anime, cartoon, and human images.
    """

    data_path = download_dataset(path=target_dir)
    print(f"[INFO] Data downloaded to {data_path}")
    
    anime_path = os.path.join(data_path, "anime")
    cartoon_path = os.path.join(data_path, "cartoon")
    human_path = os.path.join(data_path, "human")
    
    cut_data(anime_path, sample_threshold)
    cut_data(cartoon_path, sample_threshold)
    cut_data(human_path, sample_threshold)
    
    return {"anime": anime_path, "cartoon": cartoon_path, "human": human_path}
            
def save_images_and_txt(captions: List[Dict[str, str]], output_path: str = "output/samples/"):
    """
    Save a list of images to a directory.
    
    Args:
        captions (List[Dict[str, str]]): List of images to save.
        output_path (str): Path to save the images.
    """
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "captions"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
    
    caption_ls = [] 
    image_ls = []
    for i, pair in enumerate(captions):
        caption_ls.append((os.path.join(output_path, "captions", f"caption_{i}.txt"), pair["caption"]))
        
        with Image.open(pair["image"]) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            image_ls.append((os.path.join(output_path, "images", f"image_{i}.png"), img.copy()))
    
    for path, caption in tqdm(caption_ls, total=len(caption_ls), desc=f"Saving captions to {output_path}"):
        with open(path, 'w') as f:
            f.write(caption)
        
    print("[INFO] Finished saving images")
    
def save_caption_as_csv(captions: List[Dict[str, str]], output_path: str = "output"):
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
            
def save_caption_as_json(captions: List[Dict[str, str]], output_path: str = "output"):
    """
    Save a list of dictionaries to a json file.
    
    Args:
        captions (List[Dict[str, str]]): List of dictionaries to save.
        output_path (str): Path to save the json file.
    """
    os.makedirs(output_path, exist_ok=True)
    file_name = os.path.join(output_path, "captions.json")
    
    with open(file_name, 'w') as f: 
        for row in captions: 
            f.write(json.dumps(row) + '\n')
        
def save_caption_as_jsonl(captions: List[Dict[str, str]], output_path: str = "output"):
    """
    Save a list of dictionaries to a jsonl file.
    
    Args:
        captions (List[Dict[str, str]]): List of dictionaries to save.
        output_path (str): Path to save the jsonl file.
    """
    os.makedirs(output_path, exist_ok=True)
    file_name = os.path.join(output_path, "captions.jsonl")
    
    with open(file_name, 'w') as f: 
        for row in captions: 
            f.write(json.dumps(row) + '\n')
            
def check_model_exists(model_url: str):
    """
    Checks if huggingface model exists in cache.
    
    Args:
        model_url (str): Huggingface model URL.
    
    Returns:
        str: Path to the model cache.
    """
    parsed_url = model_url.split("/")
    path = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), "models--" + parsed_url[0] + "--" + parsed_url[1])
    return path

def remove_model_cache(model_url: str):
    """
    Removes the mdodel cache from the cache directory if it exists.
    
    Args:
        model_url (str): Huggingface model URL.
    """
    path = check_model_exists(model_url)
    if os.path.exists(path):
        shutil.rmtree(path)
        return True
    else:
        return False