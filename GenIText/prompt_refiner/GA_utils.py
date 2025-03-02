from typing import List, Tuple, Union, Optional
import re
from ollama import chat
from GenIText.models import LlavaModel, ViTGPT2Model, BLIPv2Model, LlavaProcessor, VITGPT2Processor, BLIPv2_Processor
import importlib.resources
import os

THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)

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
    
def get_default_config(model_id: str): 
    try: 
        with importlib.resources.path('GenITA.configs', f'{model_id}_config.yaml') as path:
            return str(path)
    except (ImportError, ModuleNotFoundError):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, 'configs', f'{model_id}_config.yaml')
