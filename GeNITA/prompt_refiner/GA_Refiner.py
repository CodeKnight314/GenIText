from ollama import chat
from PIL import Image
import random
from typing import List, Union, Dict, Tuple, Optional
from .prompts import *
from ..models import *
from ..PA_track import PerformanceTracker
import re
from glob import glob 
import os 
from tqdm import tqdm

THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)

tracker = PerformanceTracker()

@tracker.track_function
def extract_think_content(content: str) -> Tuple[str, Optional[str]]:
    """
    Extracts the clean text and think text from the LLM response content.
    
    Args:
        content: The response content from the LLM
        
    Returns:
        Tuple of (clean_text, think_text) if think text is present, otherwise just clean text
    """
    match = THINK_PATTERN.search(content)
    if match:
        think_text = match.group(0)
        clean_text = content[match.end():].strip()
        return clean_text, think_text
    return content.strip(), None

def save_prompts(prompt: Union[str, List[str]], filename: str):
    if isinstance(prompt, str):
        prompt = [prompt]
    
    with open(filename, "w") as f:
        for line in prompt:
            f.write(line + "\n")

def llm_query(
    input_content: str,
    system_context: str,
    model: str = "deepseek-r1:7b",
    deep_think: bool = False,
    print_log: bool = False
) -> Union[str, Tuple[str, str]]:
    """
    Optimized LLM query function with caching and error handling.
    
    Args:
        input_content: The input text to send to the model
        system_context: The system context for the model
        model: Model identifier (default: "llama3.2:3b")
        deep_think: Whether to return thinking process (default: False)
        print_log: Whether to print response content (default: False)
    
    Returns:
        Either clean text or tuple of (clean_text, think_text) if deep_think=True
    """
    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": input_content}
    ]
    
    try:
        response = chat(model=model, messages=messages)
        content = response["message"]["content"]
        
        if print_log:
            print(content)
        
        clean_text, think_text = extract_think_content(content)
        
        return (clean_text, think_text) if deep_think else clean_text
        
    except KeyError as e:
        raise ValueError(f"Unexpected response format from chat API: {e}")
    except Exception as e:
        raise RuntimeError(f"Error processing LLM query: {e}")

@tracker.track_function
def generate_prompt_population(prompt: str, n: int) -> List[str]:
    """
    Generates a list of n prompt variations based on the given prompt.
    
    Args:
        prompt: The base prompt to generate variations from
        n: The number of prompt variations to generate
        
    Returns:
        List of n prompt variations
    """
    
    system_prompt = """
    Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request.    
    """
    
    input_content = f"""
    Generate {n} variations of the following instruction while keep the semantic meaning. Each prompt 
    must be encompassed by <prompt> </prompt>. Write only the prompts, separated by new lines
    Input: ${prompt}
    Output:
    """
    with tracker.track_subroutine("Prompt Generation"):
        population = llm_query(input_content, system_prompt, deep_think=False)
        
    variants = []
    for line in population.strip().split("\n"):
        try:
            line = line[line.index("<prompt>") + len("<prompt>"):line.index("</prompt>")]
            variants.append(line)
        except ValueError as e: 
            continue
    
    return variants

@tracker.track_function
def choose_model(model_id: str, config: str = None):
    """
    Returns model and processor based on the model ID, loaded with the given configuration.
    
    Args:
        model_id: The model identifier
        config: The model configuration file path (default: None)
    
    Returns:
        Tuple of (model, processor) instances
    """
    models = {
            "llava": [LlavaModel, LlavaProcessor],
            "vit_gpt2": [ViTGPT2Model, VITGPT2Processor], 
            "blipv2": [BLIPv2Model, BLIPv2_Processor]
        }
    
    if model_id not in models:
        raise ValueError(f"[Error] Chosen Model ID {model_id} is not available within list of models")
    else: 
        return models[model_id][0](config), models[model_id][1](config)

@tracker.track_function
def caption_images(images: List[Image.Image], prompts: Union[List[str], str], model, processor, reranker): 
    batch = {}
    
    total = 0.0
    pbar = tqdm(prompts, desc="Scoring Prompts")
    for prompt in pbar: 
        scores = []
        for img in images:
            with tracker.track_subroutine("Image Captioning"):
                inputs = processor.preprocess(img, prompt)
                outputs = model.caption_images(inputs)
                caption = processor.postprocess(outputs)
        
            with tracker.track_subroutine("Scoring"):
                scores.append(reranker.score(img, caption))
        
        batch[prompt] = sum(scores)/len(scores)  
        total += sum(scores)/len(scores)
        pbar.set_postfix({'total_score': total})

    for key in batch.keys(): 
        batch[key] = batch[key]/total
    
    del outputs
    del inputs
    
    return batch

def choose_parents(batch: Dict):
    batch_sum = sum(list(batch.values()))
    if(batch_sum != 1.0):
        for key in batch.keys():
            batch[key] = batch[key] / batch_sum
            
    return random.choices(list(batch.keys()), weights=list(batch.values()), k=2)

@tracker.track_function
def mutate_crossover(parent_1: str, parent_2: str, context: Union[str, None] = None):
    system_context = """
    Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request. 
    """
    if context:
        system_context += f"""
        
        Generate prompts following this format style:
        Example: "{context}"
        """
    
    crossover = "Cross over the following prompts and generate a new prompt" + "\n" + parent_1 + "\n" + parent_2 + "\nEach generated prompt must be encompassed by <prompt> </prompt>"
    mutate = "Mutate the prompt and generate a final prompt bracketed with <prompt> and </prompt>"
    
    content = llm_query(crossover, system_context).strip()
    mutate = llm_query(mutate + "\n" + content, system_context).strip()
    
    mutate = mutate[mutate.index("<prompt>") + len("<prompt>"):mutate.index("</prompt>")]
    
    return mutate   

def prompt_refiner(prompt: str, 
                   image_dir: str, 
                   population_size: int,
                   generations: int, 
                   model_id: str, 
                   config: str, 
                   context: Union[str, None] = None):
    
    model, processor = choose_model(model_id, config)
    reranker = CLIPReranker()
    img_list = glob(os.path.join(image_dir, "*"))
    img_list = [Image.open(img) for img in img_list]
    img_list = [img.resize((processor.img_h, processor.img_w)) for img in img_list]
    
    os.system('cls' if os.name == 'nt' else 'clear')
    population = caption_images(img_list, generate_prompt_population(prompt, population_size), model, processor, reranker)
    pbar = tqdm(range(generations), desc="Generations")
    for gen in pbar: 
        p1, p2 = choose_parents(population)
        mutant = mutate_crossover(p1, p2, context)
        mutated_population = generate_prompt_population(mutant, population_size)
        
        m_scores = caption_images(img_list, mutated_population, model, processor, reranker)
        population = {**population, **m_scores}
        avg = sum(list(population.values()))/len(population)
        for key in population.keys():
            if(population[key] < avg) and len(population) > population_size: 
                del population[key]
        
        save_prompts(list(population.keys()), f"population_{gen}.txt")     
        os.system('cls' if os.name == 'nt' else 'clear')
        pbar.set_postfix({'avg_score': avg})
        
    population = {k: v for k, v in sorted(list(population.items()), key=lambda item: item[1], reverse=True)}
        
    return {
        "population": list(population.keys()),
        "scores": population, 
        "time": [tracker.functional_timings, tracker.subroutine_timings]
    }
        
            