from ollama import chat
from PIL import Image
import random
from typing import List, Union, Dict, Tuple
from GenIText.prompt_refiner.prompts import *
from GenIText.models import *
from GenIText.PA_track import PerformanceTracker
from GenIText.prompt_refiner.GA_utils import *
from glob import glob 
import os 
from tqdm import tqdm
from dataclasses import dataclass

tracker = PerformanceTracker()

@dataclass
class RefinerResults:
    prompt: str
    output: str
    scores: float
    raw_score: float

def postprocess_llava(output: str) -> str:
    """
    Enhanced processing for LLaVA output to ensure complete captions without unwanted line breaks.
    
    Args:
        output: Raw output from the LLaVA model
        
    Returns:
        Cleaned and completed caption as a single coherent paragraph
    """
    if not output:
        return ""
    
    if "ASSISTANT:" in output:
        output = output[output.find("ASSISTANT:") + len("ASSISTANT:"):]
    
    output = output.strip()
    output = ' '.join([line.strip() for line in output.split('\n') if line.strip()])
    output = ' '.join(output.split())
    
    if output and not output[-1] in ('.', '!', '?', ':', ';'):
        output = output + "."
    
    return output

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
    
    system_prompt = PROMPT_POPULATION_SYS
    
    input_content = PROMPT_POPULAITON_INST.format(prompt=prompt, n=n)
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
def caption_images(images: List[Image.Image], prompts: Union[List[str], str], model, processor, reranker, temperature: float = 0.5): 
    batch = {}
    
    total = 0.0
    pbar = tqdm(prompts, desc="Scoring Prompts")
    
    min_score = float('inf') 
    max_score = float('-inf')
    for prompt in pbar: 
        scores = []
        for img in images:
            with tracker.track_subroutine("Image Captioning"):
                inputs = processor.preprocess(img, prompt)
                outputs = model.caption_images(inputs)
                caption = processor.postprocess(outputs)

            caption = postprocess_llava(caption[0])
                
            with tracker.track_subroutine("Scoring"):
                scores.append(reranker.score(img, caption))
            
        prompt_score = sum(scores) / temperature
            
        batch[prompt] = RefinerResults(prompt, caption, prompt_score, prompt_score)
        total += prompt_score
        
        min_score = min(min_score, prompt_score)
        max_score = max(max_score, prompt_score)
        
        pbar.set_postfix({'Average_score': total / len(batch)})
    
    for key in batch.keys():
        batch[key].scores = (batch[key].scores - min_score) / (max_score - min_score)

    normalized_total = sum(float(batch[key].scores.item()) if hasattr(batch[key].scores, 'item') else float(batch[key].scores) for key in batch.keys())

    for key in batch.keys():
        batch[key].scores = batch[key].scores / normalized_total
    
    return batch

def choose_parents(batch: Dict):
    scores = [result.scores for result in batch.values()]
    batch_sum = sum(scores)
    
    if batch_sum != 1.0:
        for key in batch.keys():
            batch[key].scores = batch[key].scores / batch_sum
            
    return random.choices(
        list(batch.keys()), 
        weights=[result.scores for result in batch.values()], 
        k=2
    )

@tracker.track_function
def mutate_crossover(
    parent_1: str,
    parent_2: str,
    output_format: str,
    context: Union[str, None] = None
) -> str:
    """
    Combines two parent prompts and formats them according to specified output format.
    
    Args:
        parent_1: First parent prompt
        parent_2: Second parent prompt
        output_format: Desired output format specification
        context: Optional additional context
        
    Returns:
        A single mutated prompt
    """

    system_context = MUTATE_PROMPT_SYS

    if context:
        system_context += f"\nAdditional context to consider: {context}"

    crossover_instruction = CROSSOVER_PROMPT_INST.format(parent_1=parent_1, parent_2=parent_2)

    mutate_instruction = MUTATE_PROMPT_INST.format(output_format=output_format)

    merged_result = llm_query(crossover_instruction, system_context).strip()

    final_result = llm_query(
        f"{mutate_instruction}\n\nMerged Prompt:\n{merged_result}",
        system_context
    ).strip()

    if "<prompt>" in final_result and "</prompt>" in final_result:
        start_idx = final_result.index("<prompt>") + len("<prompt>")
        end_idx = final_result.index("</prompt>")
        final_prompt = final_result[start_idx:end_idx].strip()
    else:
        final_prompt = final_result

    return final_prompt  

def refiner(prompt: str, 
           image_dir: Union[str, List[str]],
           population_size: int,
           generations: int, 
           model_id: str, 
           config: str, 
           context: Union[str, None] = None):

    
    if config is None:
        config = get_default_config(model_id)
        
    model, processor = choose_model(model_id, config)
    reranker = CLIPReranker()

    if isinstance(image_dir, str):
        img_list = glob(os.path.join(image_dir, "*"))
    else:
        img_list = image_dir

    valid_img_list = []
    for img_path in img_list:
        if img_path is not None and os.path.isfile(img_path):
            valid_img_list.append(img_path)
        else:
            print("Skipping invalid path:", img_path)
            
    img_list = valid_img_list
    img_list = [Image.open(img_path) for img_path in valid_img_list]
    img_list = [img.resize((processor.img_h, processor.img_w)) for img in img_list]
    
    population = caption_images(img_list, generate_prompt_population(prompt, population_size), model, processor, reranker)
    
    try:
        pbar = tqdm(range(generations), desc="Generations")
        for gen in pbar: 
            p1, p2 = choose_parents(population)
            mutant = mutate_crossover(p1, p2, context)
            mutated_population = generate_prompt_population(mutant, population_size)
            
            m_scores = caption_images(img_list, mutated_population, model, processor, reranker)
            population = {**population, **m_scores}
            avg = sum(item.raw_score for item in population.values()) / len(population)
            
            keys_to_remove = []
            for key in population.keys():
                if(population[key].raw_score < avg) and len(population) > population_size: 
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                if len(population) > population_size:
                    del population[key]
                    
            pop_total = sum(item.score for item in population.values())
            for key in population.keys():
                population[key].scores = population[key].scores / pop_total
            
            save_prompts(list(population.keys()), f"population_{gen}.txt")     
            pbar.set_postfix({'avg_score': avg})
    
    except KeyboardInterrupt:
        return {
            "population": list(population.keys()),
            "scores": population, 
            "time": [tracker.functional_timings, tracker.subroutine_timings]
        }
            
    population = {k: v for k, v in sorted(
        list(population.items()), 
        key=lambda item: item[1].scores, 
        reverse=True
    )}
        
    return {
        "population": list(population.keys()),
        "scores": population, 
        "time": [tracker.functional_timings, tracker.subroutine_timings]
    }