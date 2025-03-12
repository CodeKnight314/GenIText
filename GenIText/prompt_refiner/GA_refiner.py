import os 
import random
import traceback
from PIL import Image
from glob import glob 
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Union, Dict

from GenIText.models import *
from GenIText.prompt_refiner.GA_utils import get_valid_image_files, color_diff
from GenIText.prompt_refiner.prompts import *
from GenIText.prompt_refiner.GA_utils import *
from GenIText.PA_track import PerformanceTracker
from GenIText.prompt_refiner.GA_LLM_Judge import llm_score

tracker = PerformanceTracker()

@dataclass
class RefinerResults:
    prompt: str
    output: str
    scores: float
    raw_score: float

@tracker.track_function
def generate_prompt_population(prompt: str, n: int, model: str = "deepseek-r1:7b") -> List[str]:
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
        population = llm_query(input_content, system_prompt, model=model, deep_think=False)
        
    variants = []
    for line in population.strip().split("\n"):
        try:
            line = line[line.index("<prompt>") + len("<prompt>"):line.index("</prompt>")]
            variants.append(line)
        except ValueError as e: 
            continue
        
    if len(variants) < n:
        variants += generate_prompt_population(prompt, n - len(variants), model)
    elif len(variants) > n:
        variants = variants[:n]
    
    return variants

@tracker.track_function
def caption_images(images: List[Image.Image], 
                   prompts: Union[List[str], str], 
                   model: BaseModel, processor: BaseModel, 
                   reranker: CLIPReranker, 
                   temperature: float = 0.5): 
    batch = {}
    
    total = 0.0
    pbar = tqdm(prompts, desc="Scoring Prompts", position=2, leave=False)
    
    if isinstance(prompts, str):
        prompts = [prompts]
            
    min_score = float('inf') 
    max_score = float('-inf')
    for prompt in pbar: 
        scores = []
        for img in images:
            try:
                with tracker.track_subroutine("Image Captioning"):
                    inputs = processor.preprocess(img, prompt)
                    outputs = model.caption_images(inputs)
                    caption = processor.postprocess(outputs)

                caption = processor.clean_string(caption[0])
                
                score = 0.0
                with tracker.track_subroutine("Scoring"):
                    r_score = reranker.score(img, caption) * 100
                    l_score = llm_score(prompt, caption, model.ollama_model)
                    score += 0.5 * l_score + 0.5 * r_score
                scores.append(score)
                
                pbar.set_postfix({"llm_score": l_score, "reranker_score": r_score})
                
            except Exception as e:
                print(f"Error processing image with prompt '{prompt[:30]}...': {str(e)}")
                traceback.print_exc()
                scores.append(0.0)
            
        prompt_score = sum(scores) / temperature
            
        batch[prompt] = RefinerResults(prompt, caption, prompt_score, prompt_score)
        total += prompt_score
        
        min_score = min(min_score, prompt_score)
        max_score = max(max_score, prompt_score)
        
        pbar.set_postfix({'Average_score': total / len(batch)})
    
    score_range = max_score - min_score
    if score_range == 0:
        for key in batch.keys():
            batch[key].scores = 1.0 / len(batch) if batch else 0
    else:
        for key in batch.keys():
            batch[key].scores = (batch[key].scores - min_score) / score_range

    normalized_total = sum(float(batch[key].scores.item()) if hasattr(batch[key].scores, 'item') else float(batch[key].scores) for key in batch.keys())

    if normalized_total > 0:
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
    context: Union[str, None] = None,
    model: str = "deepseek-r1:7b",
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

    merged_result = llm_query(crossover_instruction, system_context, model=model).strip()

    final_result = llm_query(
        f"{mutate_instruction}\n\nMerged Prompt:\n{merged_result}",
        system_context,
        model=model
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
    ollama_model = model.ollama_model
    reranker = CLIPReranker()

    if isinstance(image_dir, str):
        img_list = get_valid_image_files(image_dir)
    else:
        img_list = image_dir
            
    img_list = [Image.open(img_path) for img_path in img_list]
    img_list = [img.resize((processor.img_h, processor.img_w)) for img in img_list]
    
    initial_prompts = generate_prompt_population(prompt, population_size, ollama_model)
    
    population = caption_images(img_list, initial_prompts, model, processor, reranker)
    
    status_bar = tqdm(
        total=1, 
        bar_format="{desc}", 
        position=0,
        leave=True
    )
    
    status_bar.set_description_str(f"Current prompt: {prompt}")
    status_bar.refresh()
    
    pbar = tqdm(range(generations), desc="Generations", position=1)
    
    current_prompt = prompt
    try:
        for gen in pbar: 
            try:
                p1, p2 = choose_parents(population)
                
                mutant = mutate_crossover(p1, p2, context, ollama_model)
                mutated_population = generate_prompt_population(mutant, population_size, ollama_model)
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
                
                pop_total = sum(item.scores for item in population.values()) if population else 0
                if pop_total > 0:
                    for key in population.keys():
                        population[key].scores = population[key].scores / pop_total
                
                population = dict(sorted(population.items(), key=lambda x: x[1].scores, reverse=True))
                best_prompt = next(iter(population))
                
                current_prompt = color_diff(current_prompt, best_prompt)
                status_bar.set_description_str(f"Highest scoring prompt: {current_prompt}")
                status_bar.refresh()
                
                pbar.set_postfix({'avg_score': avg, 'population': len(population)})
                
            except Exception as e:
                print(f"Error in generation {gen}: {str(e)}")
                traceback.print_exc()
                continue
    
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