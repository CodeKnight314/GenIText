from GenITA.prompt_refiner import refiner
import argparse

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="Write a prompt that describes the image")
    parser.add_argument("--image_dir", type=str, help="path/to/image/directory")
    parser.add_argument("--config", type=str, help="path/to/config/file")
    args = parser.parse_args()
    
    prompt = refiner(
        prompt=args.prompt, 
        image_dir=args.image_dir, 
        population_size=5, 
        generations=1, 
        config=args.config,
        model_id="llava", 
        context="The response should be a single comma-separated list of keywords that describe the image."
    )
    print(prompt)