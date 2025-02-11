from GITA import End2EndCaptionPipeline, prepare_data, save_images, save_captions
import os
from typing import Dict, List
from transformers import CLIPProcessor, CLIPModel
import logging
from PIL import Image
import torch
import argparse

def calculate_similarity_score(results: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[float]]:
    """
    Calculate similarity scores between images and captions using CLIP model.
    
    Args:
        results: Dictionary mapping domains to lists of generated captions
    
    Returns:
        Dictionary mapping domains to lists of similarity scores
    """
    similarity_scores = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    
    for domain, img_caption_list in results.items():
        similarity_scores[domain] = []
        
        for img_cap_pair in img_caption_list: 
            img_path = img_cap_pair["image"]
            caption = img_cap_pair["caption"]
            
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e: 
                logging.error(f"[Error] Trouble loading image {img_path}: {e}")
                continue

            inputs = processor(text=[caption], images=img, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad(): 
                outputs = clip_model(**inputs)

            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            score = (image_embeds * text_embeds).sum(dim=-1).item()
            
            similarity_scores[domain].append(score)
    
    logging.info("=" * 50)
    for domain in similarity_scores.keys(): 
        avg_score = sum(similarity_scores[domain])/len(similarity_scores[domain])
        logging.info(f"{domain}: {avg_score:.4f}")
    logging.info("=" * 50)
    
    return similarity_scores
    
def evaluate_multi_domain_captioning(
    model_name: str,
    config_path: str, 
    output_path: str,
    threshold: int = 100
) -> Dict[str, List[str]]:
    """
    Evaluates standard image captioning performance across multiple domains (anime, cartoon, human) from
    testing dataset.
    
    Args:
        model_name: Name of the captioning model to use
        config_path: Path to model configuration file
        output_path: Directory to save results
        domains: List of domains to evaluate. If None, evaluates all domains
        threshold: Number of samples to keep for each domain when evaluating
        
    Returns:
        Dictionary mapping domains to lists of generated captions
    """
    logging.info(f"Initializing captioning pipeline with model: {model_name}")
    pipeline = End2EndCaptionPipeline(model=model_name, config=config_path)
    
    domains = ["anime", "cartoon", "human"]
    
    try:
        domain_data = prepare_data(threshold)
        results = {}
        
        for domain in domains:
            logging.info(f"Generating captions for {domain} domain")
            captions = pipeline.generate_captions(domain_data[domain])
            results[domain] = captions
            
            domain_output_path = os.path.join(output_path, domain)
            os.makedirs(domain_output_path, exist_ok=True)
            
            save_images(captions, domain_output_path)
            save_captions(captions, domain_output_path)
            
            logging.info(f"Saved {domain} results to {domain_output_path}")
            
        return results
        
    except Exception as e:
        logging.error(f"Error during caption generation: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model choice for End2End pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config file for selected model.")
    parser.add_argument("--save", type=str, default="outputs/", help="Path to output path for images, captions, and txt files")
    parser.add_argument("--threshold", type=int, default=100, help="Threshold for number of testing samples to perserve.")
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    results = evaluate_multi_domain_captioning(
        model_name=args.model, 
        config_path=args.config, 
        output_path=args.save, 
        threshold=args.threshold
    )
    
    scores = calculate_similarity_score(results)