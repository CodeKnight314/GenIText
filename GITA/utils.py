import os
from tqdm import tqdm
from typing import Dict, List
import kagglehub
from random import sample
import shutil
import requests
import zipfile

def download_dataset(url: str = None, path: str = "dataset/"):
    if url is None:
        url = "https://www.kaggle.com/api/v1/datasets/download/hadiepratamatulili/anime-vs-cartoon-vs-human"

    os.makedirs(path, exist_ok=True)
    download_path = os.path.join(path, "anime-vs-cartoon-vs-human.zip")

    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with open(download_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(block_size):
                f.write(chunk)
                bar.update(len(chunk))

        print(f"[INFO] Successfully downloaded to {download_path}")

        print("[INFO] Extracting files...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(path)
            print(f"[INFO] Extracted all files to {path}")
        
        os.remove(download_path)
        return os.path.join(path, "Data/")
    else:
        print(f"Download failed with status code: {response.status_code}")

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
    Prepare the dataset for training.
    
    Args:
        sample_threshold (int): Number of samples to keep.
        
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
    
    return anime_path, cartoon_path, human_path
            
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