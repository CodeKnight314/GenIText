from GenITA.prompt_refiner import prompt_refinement
import argparse

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="Write a prompt that describes the image")
    parser.add_argument("--image_dir", type=str, help="path/to/image/directory")
    parser.add_argument("--config", type=str, help="path/to/config/file")
    args = parser.parse_args()
    
    prompt = prompt_refinement(
        prompt=args.prompt, 
        image_dir=args.image_dir, 
        population_size=5, 
        generations=5, 
        config=args.config,
        model_id="llava"
    )
    print(prompt)