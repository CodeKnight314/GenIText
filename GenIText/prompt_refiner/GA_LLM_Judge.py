from typing import Dict, Union, Tuple, List, Optional
import re
from GenIText.prompt_refiner.GA_utils import llm_query

def llm_score(
    caption: str, 
    prompt: str, 
    model: str = "deepseek-r1:1.5b",
    deep_think: bool = False
) -> float:
    """
    Score a caption based on how well it matches the given prompt and context.
    Returns an aggregate score calculated from individual criteria scores.
    
    Args:
        caption: The caption to evaluate
        prompt: The original prompt that generated the caption
        context: Optional additional instructions or constraints
        criteria: Optional dictionary of scoring criteria and their weights
                 Default criteria: relevance (0.4), accuracy (0.3), coherence (0.2), creativity (0.1)
        model: Model to use for evaluation
        deep_think: Whether to return thinking process
        
    Returns:
        A single float score representing the quality of the caption
    """
    criteria = {
        "relevance": 0.4,
        "accuracy": 0.3, 
        "coherence": 0.2,
        "creativity": 0.1
    }
    
    system_context = f"""
    You are an expert caption evaluator. Your task is to evaluate how well a caption matches its prompt.
    
    SCORING CRITERIA:
    - Relevance: How directly relevant the caption is to the prompt's subject matter.
    - Accuracy: How accurately the caption reflects the prompt's intent and details.
    - Coherence: How well-structured, clear, and coherent the caption is.
    - Creativity: How creative, engaging, or original the caption is.
    
    Evaluate the caption on a scale of 0-100 for each criterion.
    
    IMPORTANT: Your output MUST follow this exact format, with one score per line:
    Relevance: [score]
    Accuracy: [score]
    Coherence: [score]
    Creativity: [score]
    
    Just return the scores with no additional explanation or text.
    """
    
    input_content = f"""
    PROMPT: {prompt}
    
    CAPTION: {caption}
    
    Evaluate this caption according to the criteria.
    Remember to follow the exact output format:
    
    Relevance: [score]
    Accuracy: [score]
    Coherence: [score]
    Creativity: [score]
    """
    
    result = llm_query(
        input_content=input_content,
        system_context=system_context,
        model=model,
        deep_think=deep_think
    )
    
    if deep_think:
        response, thinking = result
    else:
        response = result
    
    try:
        cleaned_response = response.strip()
        
        if "```" in cleaned_response:
            code_blocks = re.findall(r'```(?:\w*\n)?(.*?)```', cleaned_response, re.DOTALL)
            if code_blocks:
                cleaned_response = code_blocks[0].strip()
        
        extracted_scores = {}
        
        criteria_patterns = {
            "relevance": r'relevance:?\s*(\d+(?:\.\d+)?)',
            "accuracy": r'accuracy:?\s*(\d+(?:\.\d+)?)',
            "coherence": r'coherence:?\s*(\d+(?:\.\d+)?)',
            "creativity": r'creativity:?\s*(\d+(?:\.\d+)?)'
        }
        
        for criterion, pattern in criteria_patterns.items():
            match = re.search(pattern, cleaned_response, re.IGNORECASE)
            if match:
                extracted_scores[criterion] = float(match.group(1))
        
        if not extracted_scores:
            for criterion in criteria.keys():
                pattern = fr'{criterion}\s*-\s*(\d+(?:\.\d+)?)'
                match = re.search(pattern, cleaned_response, re.IGNORECASE)
                if match:
                    extracted_scores[criterion] = float(match.group(1))
        
        if not extracted_scores:
            lines = cleaned_response.split('\n')
            score_lines = [line for line in lines if re.search(r'\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?', line)]
            
            if len(score_lines) >= len(criteria):
                for i, criterion in enumerate(criteria.keys()):
                    if i < len(score_lines):
                        fraction_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*\d+(?:\.\d+)?', score_lines[i])
                        if fraction_match:
                            extracted_scores[criterion] = float(fraction_match.group(1))
        
        if not extracted_scores:
            numbers = re.findall(r'(\d+(?:\.\d+)?)', cleaned_response)
            if len(numbers) >= len(criteria):
                for i, criterion in enumerate(criteria.keys()):
                    if i < len(numbers):
                        extracted_scores[criterion] = float(numbers[i])
        
        if len(extracted_scores) >= len(criteria):
            weighted_score = 0
            for criterion, weight in criteria.items():
                if criterion in extracted_scores:
                    weighted_score += extracted_scores[criterion] * weight
            
            weighted_score = max(0, min(weighted_score, 100))
            return weighted_score
        else:
            return -1.0
            
    except Exception as e:
        return -1.0