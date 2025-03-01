PROMPT_POPULATION_SYS = """
    Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request.    
    """
    
PROMPT_POPULAITON_INST = """
    Generate {n} variations of the following instruction while keep the semantic meaning. Each prompt 
    must be always be encompassed by <prompt> </prompt>. Write only the prompts, separated by new lines
    Input: {prompt}
    Output:
    """
    
MUTATE_PROMPT_SYS = """
    You will combine (cross over) the two provided instructions into a single new prompt.
    Then you will mutate that new prompt so that it explicitly directs the user to produce
    output in the style/format given by 'output_format'.
    
    Your final response should ONLY contain the newly mutated prompt wrapped as:
    <prompt> FINAL_INSTRUCTION </prompt>
    """
    
CROSSOVER_PROMPT_INST = """
    Combine these two instructions into a single cohesive prompt:
    1) {parent_1}
    2) {parent_2}
    Preserve the intent of prompt while combining both prompts.
    """
    
MUTATE_PROMPT_INST = """
    Now mutate this merged prompt so that it explicitly instructs the user/model
    to produce the final output according to the following format guidelines:
    {output_format}

    Wrap only the final mutated prompt in <prompt>...</prompt> and nothing else.
    """