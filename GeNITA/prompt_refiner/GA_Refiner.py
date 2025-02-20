from ollama import chat
from PIL import Image
import random
from typing import List, Union, Dict, Tuple, Optional
from .prompts import *
from ..models import *
import re
from glob import glob 
import os 
from tqdm import tqdm 

THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)

def extract_think_content(content: str) -> Tuple[str, Optional[str]]:
    """Extract thinking content and clean text using regex."""
    match = THINK_PATTERN.search(content)
    if match:
        think_text = match.group(0)
        clean_text = content[match.end():].strip()
        return clean_text, think_text
    return content.strip(), None

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

def generate_prompt_population(prompt: str, n: int) -> List[str]:
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
    population = llm_query(input_content, system_prompt, deep_think=False)
    variants = []
    for line in population.strip().split("\n"):
        try:
            line = line[line.index("<prompt>") + len("<prompt>"):line.index("</prompt>")]
            variants.append(line)
        except ValueError as e: 
            continue
    
    return variants

def choose_model(model_id: str, config: str = None):
    models = {
            "llava": [LlavaModel, LlavaProcessor],
            "vit_gpt2": [ViTGPT2Model, VITGPT2Processor], 
            "blipv2": [BLIPv2Model, BLIPv2_Processor]
        }
    
    if model_id not in models:
        raise ValueError(f"[Error] Chosen Model ID {model_id} is not available within list of models")
    else: 
        return models[model_id][0](config), models[model_id][1](config)

def caption_images(images: List[Image.Image], prompts: Union[List[str], str], model, processor, reranker): 
    batch = {}
    
    total = 0.0
    pbar = tqdm(prompts, desc="Scoring Prompts")
    for prompt in pbar: 
        scores = []
        for img in images: 
            inputs = processor.preprocess(img, prompt)
            outputs = model.caption_images(inputs)
            caption = processor.postprocess(outputs)
            
            scores.append(reranker.score(img, caption))
            
        batch[prompt] = sum(scores)/len(scores)  
        total += sum(scores)/len(scores)
        pbar.set_postfix({'total_score': total})

    for key in batch.keys(): 
        batch[key] = batch[key]/total
    
    gen_avg = total / (len(prompts) * len(images))
    
    del outputs
    del inputs
    
    return batch, gen_avg

def choose_parents(batch: Dict):
    return random.choices(list(batch.keys()), weights=list(batch.values()), k=2)

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
       
def prompt_refinement(prompt: str, 
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
    
    population = generate_prompt_population(prompt, population_size)
    pbar = tqdm(range(generations), desc="Generations")
    for gen in pbar:
        scores, gen_avg = caption_images(img_list, population, model, processor, reranker)
        pbar.set_postfix({'best_score': max(scores.values()), 'avg_score': gen_avg})
        scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        scores = dict(list(scores.items())[:population_size])
        population = list(scores.keys())
        
        parent_1, parent_2 = choose_parents(scores)
        mutant = mutate_crossover(parent_1, parent_2, context)
        mutated_population = generate_prompt_population(mutant, population_size)
        population.extend(mutated_population)
        
    return list(scores.keys())[0], scores[list(scores.keys())[0]]   