import torch 
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Union
from tqdm import tqdm
import numpy as np

class CLIPReranker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_image(self, image: Union[Image.Image, str, List[Image.Image]]) -> Image.Image:
        """Helper method to load and validate images."""
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to load image: {e}")
        elif isinstance(image, list):
            for i, img in enumerate(image):
                if isinstance(img, str):
                    try:
                        image[i] = Image.open(img).convert("RGB")
                    except Exception as e:
                        raise ValueError(f"Failed to load image {i}: {e}")
                elif not isinstance(img, Image.Image):
                    raise ValueError(f"Invalid image type at index {i}")
        elif not isinstance(image, Image.Image):
            raise ValueError("Invalid image type")
        
        return image

    def score(self, image: Union[Image.Image, str, List[Image.Image]], captions: Union[List[str], str]) -> np.ndarray:
        """
        Calculate CLIP similarity scores between an image and caption(s).
        
        Args:
            image: PIL Image or path to image
            captions: Single caption string or list of caption strings
            
        Returns:
            numpy array of similarity scores
        """
        image = self._load_image(image)
        if isinstance(captions, str):
            captions = [captions]
            
        try:
            inputs = self.processor(
                text=captions,
                images=image,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                
                similarity = (image_embeds @ text_embeds.T).cpu().numpy()
                
                scores = (similarity + 1) / 2
                if len(captions) == 1:
                    scores = scores.flatten()
                
            return scores
            
        except Exception as e:
            raise RuntimeError(f"Error computing CLIP scores: {e}")

    def rerank(
        self,
        images: Union[List[Image.Image], List[str]],
        captions: Union[List[List[str]], List[str]]
    ) -> List[dict]:
        """
        Rerank images based on CLIP similarity scores with captions.
        
        Args:
            images: List of PIL Images or image paths
            captions: List of caption strings or list of caption lists
            
        Returns:
            List of dicts containing reranking results for each image
        """
        if not images or not captions:
            raise ValueError("Empty images or captions list")
        if len(images) != len(captions):
            raise ValueError("Number of images and caption sets must match")
            
        results = []
        for i, image in enumerate(tqdm(images, desc="Scoring images")):
            try:
                current_captions = captions[i] if isinstance(captions[i], list) else [captions[i]]
                scores = self.score(image, current_captions)
                
                result = {
                    'image_idx': i,
                    'image': image,
                    'best_caption': current_captions[scores.argmax()],
                    'best_score': float(scores.max()),
                    'all_scores': scores.tolist()
                }
                results.append(result)
                
            except Exception as e:
                print(f"Warning: Failed to process image {i}: {e}")
                continue
                
        return results