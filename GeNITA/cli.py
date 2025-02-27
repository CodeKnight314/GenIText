import click
import os
import shlex
from GenITA.pipelines import End2EndCaptionPipeline
from GenITA.prompt_refiner import refiner
from GenITA.utils import save_images, save_captions

def title_screen(): 
    os.system("clear" if os.name == "posix" else "cls")
    click.echo(click.style("\n ██████╗ ███████╗███╗   ██╗██╗████████╗ █████╗ ", fg="red", bold=True))
    click.echo(click.style("██╔════╝ ██╔════╝████╗  ██║██║╚══██╔══╝██╔══██╗", fg="red"))
    click.echo(click.style("██║  ███╗█████╗  ██╔██╗ ██║██║   ██║   ███████║", fg="red"))
    click.echo(click.style("██║   ██║██╔══╝  ██║╚██╗██║██║   ██║   ██╔══██║", fg="red"))
    click.echo(click.style("╚██████╔╝███████╗██║ ╚████║██║   ██║   ██║  ██║", fg="red", bold=True))
    click.echo(click.style(" ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝  ╚═╝", fg="red"))
    
    click.echo("\nWelcome to GENITA! This package is designed to generate captions for a list of images using an End2End pipeline.")
    click.echo("Type '/help' to see the available commands.")

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx): 
    """GenITA: General Image-to-Text Automated package"""
    if ctx.invoked_subcommand is None: 
        start_interactive_shell()

def show_help(): 
    click.echo("\nAvailable commands:")
    click.echo("/caption <image_path/image_folder> --model <model_name> --output <output_path> --config <config>")
    click.echo("/refine <prompt> <image_path/image_folder> <context> --model <model_name> --config <config> --output <output_path>")
    click.echo("/models - Show available models")
    click.echo("/help - Show this help menu")
    click.echo("/clear - Clear the screen")
    click.echo("/exit - Exit GENITA")

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
@click.option("--model", "-m", default="vit-gpt2", help="Model name to use for captioning.")
@click.option("--output", "-o", default="output/", help="Output directory.")
@click.option("--config", "-c", help="Configuration file.")        
def caption(image_path: str, model_name: str, output_path: str):
    """
    Generate captions for a list of images.
    """
    if os.path.isfile(image_path):
        image_paths = [image_path]
    elif os.path.isdir(image_path):
        image_paths = [os.path.join(image_path, img) for img in os.listdir(image_path)]
    else:
        raise FileNotFoundError(f"[ERROR] {image_path} not found.")
    
    click.echo(f"[INFO] Generating captions for {len(image_paths)} images using {model_name} model")
    pipeline = End2EndCaptionPipeline(model=model_name, config=None)
    
    captions = pipeline.generate_captions(image_paths)
    os.makedirs(output_path, exist_ok=True)
    
    save_images(captions, output_path)
    save_captions(captions, output_path)
    
    click.echo(f"[INFO] Captions saved to {output_path}")

@cli.command()
@click.argument("prompt")
@click.argument("image_dir", type=click.Path(exists=True))
@click.argument("context")
@click.option("--model", "-m", default="llava", help="Model to use for refinement.")
@click.option("--config", "-c", help="Configuration file.")
def refine(prompt: str, image_dir: str, context: str, model_id: str = "llava", config: str = None):
    """
    Refine a prompt to generate a better caption.
    """
    if os.path.isfile(image_dir):
        image_paths = [image_dir]
    elif os.path.isdir(image_dir):
        image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
    else:
        raise FileNotFoundError(f"[ERROR] {image_dir} not found.")
    
    click.echo(f"[INFO] Refining prompt for {len(image_paths)} images using {model_id} model")
    refined_prompt = refiner(
        prompt=prompt, 
        image_dir=image_paths, 
        population_size=5, 
        generations=5, 
        config=config,
        model_id=model_id, 
        context=context
    )
    
    optimal_prompt = refined_prompt["population"][0]
    optimal_score = refined_prompt["scores"][0]
    
    click.echo(f"[INFO] Refined prompt: \n{optimal_prompt} \n[INFO] Score: {optimal_score}")

def start_interactive_shell(): 
    os.system('clear' if os.name == 'posix' else 'cls')
    title_screen()
    
    command_map = {
        '/caption': caption,
        '/refine': refine,
        '/models': models,
    }
    
    while True: 
        try: 
            command = input(click.style("\nGenITA> ", fg="red", bold=True)).strip()
            
            if command == "/help": 
                show_help()
            elif command == "/exit": 
                click.echo(click.style("\n[INFO] Exiting GENITA", fg="red"))
                break
            elif command == "/clear":
                os.system('clear' if os.name == 'posix' else 'cls')
                title_screen()
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
                        click.echo(click.style(f"[ERROR] {e}", fg="red"))
                        click.echo(cmd.get_help(click.Context(cmd)))    
                pass
            else:
                click.echo(click.style("[ERROR] Invalid command. Type '/help' to see the available commands.", fg="red"))
        except KeyboardInterrupt:
            click.echo(click.style("\n[INFO] Exiting GENITA", fg="red"))
            break
    
if __name__ == "__main__":
    cli()