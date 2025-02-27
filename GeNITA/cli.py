import click
import os
from .pipelines import End2EndCaptionPipeline

def title_screen(): 
    os.system("clear" if os.name == "posix" else "cls")
    click.echo(click.style("\n ██████╗ ███████╗███╗   ██╗██╗████████╗ █████╗ ", fg="red", bold=True))
    click.echo(click.style("██╔════╝ ██╔════╝████╗  ██║██║╚══██╔══╝██╔══██╗", fg="red"))
    click.echo(click.style("██║  ███╗█████╗  ██╔██╗ ██║██║   ██║   ███████║", fg="red"))
    click.echo(click.style("██║   ██║██╔══╝  ██║╚██╗██║██║   ██║   ██╔══██║", fg="red"))
    click.echo(click.style("╚██████╔╝███████╗██║ ╚████║██║   ██║   ██║  ██║", fg="red", bold=True))
    click.echo(click.style(" ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝  ╚═╝", fg="red"))
    
    click.echo("\nWelcome to GenITA! This package is designed to generate captions for a list of images using an End2End pipeline.")
    click.echo("The pipeline consists of a pre-trained image captioning model and a pre-trained image classification model.")
    click.echo("Type 'genita --help' to see the available commands.\n")

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx): 
    """GenITA: General Image-to-Text Automated package"""
    start_interactive_shell()

def show_help(): 
    os.system('clear' if os.name == 'posix' else 'cls')
    title_screen()
    click.echo("Available commands:")
    click.echo("/caption <image_path/image_folder> --model <model_name> --output <output_path>")
    click.echo("/refine <image_path/image_folder> <n_iteration> <population_size> --model <model_name> --output <output_path>")
    click.echo("/help - Show this help menu")
    click.echo("/exit - Exit GENITA")
    
def start_interactive_shell(): 
    os.system('clear' if os.name == 'posix' else 'cls')
    title_screen()
    while True: 
        try: 
            command = input(click.style("\nGenITA> ", fg="red", bold=True)).strip()
            if command == "/help": 
                show_help()
            elif command == "/exit": 
                click.echo(click.style("\n[INFO] Exiting GENITA", fg="red"))
                break
            elif command.startswith("/caption"): 
                pass
            elif command.startswith("/refine"):
                pass
            else: 
                click.echo(click.style("Invalid command. Type '/help' to see the available commands.", fg="red"))
        except KeyboardInterrupt:
            click.echo(click.style("\n[INFO] Exiting GENITA", fg="red"))
            break
        
if __name__ == "__main__":
    cli()