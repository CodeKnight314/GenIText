from setuptools import setup, find_packages

setup(
    name="genitext",
    version="0.3.1",
    author="Richard Tang",
    author_email="richardgtang@gmail.com",
    description="A CLI tool for generating high-quality image-text pairs for AI training",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CodeKnight314/GenIText",
    packages=find_packages(include=["GenIText", "GenIText.*"]),
    include_package_data=True,
    package_data={
        "GenIText": ["configs/*.yaml"],
    },
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
        "click"
    ],
    entry_points={
        "console_scripts": [
            "genitext=GenIText.cli:cli",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    )
