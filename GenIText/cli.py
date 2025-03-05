import click
import os
import shlex
import traceback
import warnings
from transformers import logging
import importlib.resources
import yaml

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit import PromptSession 
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.document import Document

from GenIText.pipelines import End2EndCaptionPipeline
from GenIText.prompt_refiner import refiner
from GenIText.utils import *
from GenIText.config_editor import ConfigEditor

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

class CommandAuotSuggest(AutoSuggest):
    def __init__(self, commands):
        self.commands = commands
    
    def get_suggestion(self, buffer, document):
        text = document.text_before_cursor
        for cmd in self.commands:
            if cmd.startswith(text) and cmd != text:
                return Suggestion(cmd[len(text):])
        return None

class PathAndOptionsAutoSuggest(AutoSuggest):
    def __init__(self):
        self.dir = os.getcwd()
        self.dir_list = os.listdir(self.dir)
        self.options = {
            "/caption": ["--model", "--output", "-m", "-o"],
            "/refine": ["--model", "--pop", "--gen", "-m", "-p", "-g"],
        }
        
    def get_suggestion(self, command, buffer, document):
        text = document.text_before_cursor
        for file in self.dir_list:
            if file.startswith(text):
                return Suggestion(file[len(text):])
        
        if command in self.options:
            for option in self.options[command]:
                if option.startswith(text):
                    return Suggestion(option[len(text):])
        return None

class InterfaceAutoSuggest(AutoSuggest):
    def __init__(self, commands):
        self.command_suggestor = CommandAuotSuggest(commands)
        self.path_suggestor = PathAndOptionsAutoSuggest()
    
    def get_suggestion(self, buffer, document):
        tokens = document.text_before_cursor.split()
        
        if not tokens:
            return None
        
        if len(tokens) == 1:
            return self.command_suggestor.get_suggestion(buffer, document)
        else:
            last_token = tokens[-1]
            dummy_doc = Document(text=last_token, cursor_position=len(last_token))
            return self.path_suggestor.get_suggestion(tokens[0], buffer, dummy_doc)

def title_screen(): 
    os.system("clear" if os.name == "posix" else "cls")
    click.echo(click.style("\n ██████╗ ███████╗███╗   ██╗██╗████████╗███████╗██╗  ██╗████████╗", fg="red", bold=True))
    click.echo(click.style("██╔════╝ ██╔════╝████╗  ██║██║╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝", fg="red", bold=True))
    click.echo(click.style("██║  ███╗█████╗  ██╔██╗ ██║██║   ██║   █████╗   ╚███╔╝    ██║   ", fg="red", bold=True))
    click.echo(click.style("██║   ██║██╔══╝  ██║╚██╗██║██║   ██║   ██╔══╝   ██╔██╗    ██║   ", fg="red", bold=True))
    click.echo(click.style("╚██████╔╝███████╗██║ ╚████║██║   ██║   ███████╗██╔╝ ██╗   ██║   ", fg="red", bold=True))
    click.echo(click.style(" ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   ", fg="red", bold=True))
    
    click.echo("\nWelcome to GENITEXT v0.3.1! This package is designed to generate captions for a list of images using an End2End pipeline.")
    click.echo("Type '/help' to see the available commands.")

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx): 
    """GenIText: General Image-to-Text Automated package"""
    if ctx.invoked_subcommand is None: 
        start_interactive_shell()

def show_help(): 
    click.echo("\nAvailable commands:")
    click.echo("/caption <image_path/image_folder> --model <model_name> --output <output_path>")
    click.echo("/refine <prompt> <image_path/image_folder> <context> --model <model_name> --pop <population_size> --gen <generations>")
    click.echo("/delete <model_name> - Delete a model")
    click.echo("/ls - List files in the current directory")
    click.echo("/models - Show available models")
    click.echo("/config <model_name> - Modify model configs")
    click.echo("/help - Show this help menu")
    click.echo("/clear - Clear the screen")
    click.echo("/exit - Exit GenIText")

@cli.command()
def models():
    """
    Show available models for captioning.
    """
    models = End2EndCaptionPipeline.models
    click.echo("Available models:")
    for model in models:
        click.echo(f"- {model}")
        
@cli.command() 
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--model", "-m", default="vit_gpt2", type=click.Choice(list(End2EndCaptionPipeline.models.keys())), help="Model name to use for captioning.")
@click.option("--output", "-o", default="output/", help="Output directory.") 
@click.option("--format", "-f", default="json", type=click.Choice(['json', 'jsonl', 'csv', 'img&txt'], case_sensitive=False), help="Output format (json/jsonl/csv/img&txt).")       
def caption(image_path: str, model: str, output: str, format: str):
    """
    Generate captions for a list of images.
    """
    if os.path.isfile(image_path):
        image_paths = [image_path]
    elif os.path.isdir(image_path):
        image_paths = [os.path.join(image_path, img) for img in os.listdir(image_path)]
    else:
        raise FileNotFoundError(f"[ERROR] {image_path} not found.")
    
    click.echo(f"[INFO] Generating captions for {len(image_paths)} images using {model} model")
    pipeline = End2EndCaptionPipeline(model=model, config=None)
    
    captions = pipeline.generate_captions(image_paths)
    os.makedirs(output, exist_ok=True)
    
    if format == "json":
        save_caption_as_json(captions, output)
    elif format == "jsonl":
        save_caption_as_jsonl(captions, output)
    elif format == "csv":
        save_caption_as_csv(captions, output)
    elif format == "img&txt":
        save_images_and_txt(captions, output)
    else: 
        raise ValueError(f"[ERROR] Invalid format: {format}")
    
    click.echo(f"[INFO] Captions saved to {output}")
    
@cli.command()
@click.argument("model", default="llava", type=click.Choice(list(End2EndCaptionPipeline.models.keys())))
def delete(model: str):
    """
    Delete a model.
    """
    with importlib.resources.path('GenIText.configs', f'{model}_config.yaml') as path:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            model_url = config["model"]["model_id"]
            if(remove_model_cache(model_url)): 
                click.echo(f"[INFO] Model {model} deleted.")
            else:
                click.echo(f"[ERROR] Model {model} not found.")
                
@cli.command() 
@click.argument("model", default="llava", type=click.Choice(list(End2EndCaptionPipeline.models.keys())))         
def config(model: str):
    """
    Modify configs
    """
    with importlib.resources.path('GenIText.configs', f'{model}_config.yaml') as path:
        editor = ConfigEditor(config=path, model=model)
        editor.run()

@cli.command()
@click.argument("prompt")
@click.argument("image_dir", type=click.Path(exists=True))
@click.argument("context")
@click.option("--model", "-m", default="llava", type=click.Choice(list(End2EndCaptionPipeline.models.keys())), help="Model to use for refinement.")
@click.option("--pop", "-p", default=5, help="Population size for refinement.")
@click.option("--gen", "-g", default=5, help="Number of generations for refinement.")
def refine(prompt: str, image_dir: str, context: str, model: str = "llava", pop: int = 5, gen: int = 5):
    """
    Refine a prompt to generate a better caption.
    """
    click.echo(f"[INFO] Starting refine with image_dir: {image_dir}")
        
    if os.path.isfile(image_dir):
        image_paths = [image_dir]
        click.echo(f"[INFO] Single file mode: {image_paths}")
    elif os.path.isdir(image_dir):
        image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        click.echo(f"[INFO] Directory mode: Found {len(image_paths)} files")
    else:
        raise FileNotFoundError(f"[ERROR] {image_dir} not found.")
    
    click.echo(f"[INFO] Refining prompt for {len(image_paths)} images using {model} model")
    for i, path in enumerate(image_paths):
        if path is None:
            click.echo(f"[WARNING] Path {i} is None!")
        elif not os.path.exists(path):
            click.echo(f"[WARNING] Path {i} does not exist: {path}")
        

    click.echo(f"[INFO] Model: {model}, Config: Default, Population: {pop}, Generations: {gen}")
    
    refined_prompt = refiner(
        prompt=prompt, 
        image_dir=image_paths, 
        population_size=pop, 
        generations=gen, 
        config=None,
        model_id=model, 
        context=context
    )
    
    optimal_prompt = refined_prompt["population"][0]
    optimal_score = refined_prompt["scores"][0]
    
    click.echo(f"[INFO] Refined prompt: \n{optimal_prompt} \n[INFO] Score: {optimal_score}")

def start_interactive_shell(): 
    os.system('clear' if os.name == 'posix' else 'cls')
    title_screen()
    
    bindings = KeyBindings()
    @bindings.add("tab")
    def accept_auto_suggestion(event):
        """
        When Tab is pressed, check if there's an auto-suggestion available.
        If yes, insert the suggestion text into the buffer.
        Otherwise, you can trigger the default completion behavior.
        """
        buff = event.current_buffer
        if buff.suggestion:
            buff.insert_text(buff.suggestion.text)
        else:
            event.app.current_buffer.start_completion(select_first=False)
    
    command_map = {
        '/caption': caption,
        '/refine': refine,
        '/models': models,
        '/help': show_help,
        '/delete': delete,
        '/config': config,
        '/ls': None,
        '/clear': None,
        '/exit': None
    }
    
    session = PromptSession(auto_suggest=InterfaceAutoSuggest(list(command_map.keys())),key_bindings=bindings)
    while True: 
        try: 
            command = session.prompt(f"\n~/GenIText> ")
            
            if command == "/help": 
                show_help()
            elif command == "/exit": 
                click.echo(click.style("\n[INFO] Exiting GenIText", fg="red"))
                break
            elif command == "/clear":
                os.system('clear' if os.name == 'posix' else 'cls')
                title_screen()
                
            elif command == "/ls":
                current_dir = os.getcwd()
                click.echo(f"Current directory: {current_dir}")
                for file in os.listdir(current_dir):
                    click.echo(f"- {file}")
            elif command.startswith(tuple(command_map.keys())):
                parts = shlex.split(command)
                cmd = command_map.get(parts[0])
                
                if cmd is None: 
                    click.echo(click.style("[ERROR] Invalid command. Type '/help' to see the available commands.", fg="red"))
                else: 
                    args = parts[1:]
                    try: 
                        cmd.main(args=args, standalone_mode=False)
                    except Exception as e:
                        traceback.print_exc()
                        click.echo(click.style(f"[ERROR] {e}", fg="red"))
                        click.echo(cmd.get_help(click.Context(cmd)))    
                pass
            else:
                click.echo(click.style("[ERROR] Invalid command. Type '/help' to see the available commands.", fg="red"))
        except KeyboardInterrupt:
            click.echo(click.style("\n[INFO] Exiting GenIText", fg="red"))
            break
    
if __name__ == "__main__":
    cli()