from ollama import chat
from PIL import Image
import random
from typing import List, Union, Dict, Tuple, Optional
from GenITA.prompt_refiner.prompts import *
from GenITA.models import *
from GenITA.PA_track import PerformanceTracker, TimingStats
import re
from glob import glob 
import os 
from tqdm import tqdm
from dataclasses import dataclass

THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)

tracker = PerformanceTracker()

@dataclass
class RefinerResults:
    prompt: str
    output: str
    scores: float

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
    must be always be encompassed by <prompt> </prompt>. Write only the prompts, separated by new lines
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
            
        batch[prompt] = RefinerResults(prompt, caption, prompt_score)
        total += prompt_score
        
        min_score = min(min_score, prompt_score)
        max_score = max(max_score, prompt_score)
        
        pbar.set_postfix({'Average_score': total / len(batch)})
    
    for key in batch.keys():
        batch[key].scores = (batch[key].scores - min_score) / (max_score - min_score)

    normalized_total = sum(float(batch[key].scores.item()) if hasattr(batch[key].scores, 'item') else float(batch[key].scores) for key in batch.keys())

    for key in batch.keys():
        batch[key].scores = batch[key].scores / normalized_total
        
    print(f"Min Score: {min_score}, Max Score: {max_score}")
    print("Distribution: ", [result.scores for result in batch.values()])
    
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
    Crosses over two prompt instructions (parent_1, parent_2) and mutates them
    into a single prompt that instructs the user/model to produce output
    according to a user-specified 'output_format'.

    Parameters:
    - parent_1: First prompt instruction.
    - parent_2: Second prompt instruction.
    - output_format: A free-form string describing the format you want
      the final output (i.e., how to instruct the LLM to respond).
      Example: "The response should be a single comma-separated list of keywords."
      or "Use a fully descriptive sentence with minimal punctuation."

    - context (optional): Extra context or style info you want to pass along.

    Returns:
    - A single mutated prompt (str), without <prompt>...</prompt> tags, which
      can be used downstream.
    """

    system_context = """
    You will combine (cross over) the two provided instructions into a single new prompt.
    Then you will mutate that new prompt so that it explicitly directs the user to produce
    output in the style/format given by 'output_format'.
    
    Your final response should ONLY contain the newly mutated prompt wrapped as:
    <prompt> FINAL_INSTRUCTION </prompt>
    """

    if context:
        system_context += f"\nAdditional context to consider: {context}"

    crossover_instruction = f"""
    Combine these two instructions into a single cohesive prompt:
    1) {parent_1}
    2) {parent_2}
    Preserve the intent of prompt while combining both prompts.
    """

    mutate_instruction = f"""
    Now mutate this merged prompt so that it explicitly instructs the user/model
    to produce the final output according to the following format guidelines:
    {output_format}

    Wrap only the final mutated prompt in <prompt>...</prompt> and nothing else.
    """

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
        avg = sum(item.scores for item in population.values()) / len(population)
        
        keys_to_remove = []
        for key in population.keys():
            if(population[key].scores < avg) and len(population) > population_size: 
                keys_to_remove.append(key)

        for key in keys_to_remove:
            if len(population) > population_size:
                del population[key]
        
        save_prompts(list(population.keys()), f"population_{gen}.txt")     
        os.system('cls' if os.name == 'nt' else 'clear')
        pbar.set_postfix({'avg_score': avg})
        
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