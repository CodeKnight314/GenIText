from setuptools import setup, find_packages

setup(
    name="GenITA",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch", 
        "torchvision", 
        "Pillow", 
        "tqdm", 
        "matplotlib",
        "pyyaml", 
        "bitsandbytes", 
        "accelerate", 
        "numpy", 
        "transformers", 
        "typing-extensions", 
        "ollama",
        "click"],
    entry_points={
        "console_scripts": [
            "genita=GenITA.cli:cli"
        ]
    },
    author="Richard Tang",
    author_email="richardgtang@gmail.com"
)