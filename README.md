# GenIText: Generative Image-Text Automated package

<p align="center">
  <img src="resources/demo.gif" alt="Demonstration video of GenIText tool">
</p>

## Overview
This repository is independently developed as a felxible framework to generate high-quality Image-Text pairs for finetuning Image-Generation models, such as Stable Diffusion, DALL-E, and other generative models. By leveraging open-source captioning models, GenIText automates the process of generating diverse captions for corresponding images, ensuring that the text data is well-suited for downstream applications such as style-specific generations or domain adaptation. This framework is designed to complement contemporary repositories or modules in the field, offering an additional option for flexibility and automation to create customized datasets.

GenIText will become distributable as a CLI tool once package is ready for testing across systems. Please support in any way you see fit!

## Table of Contents
- [Installation](#installation)
- [Benchmarks](#Benchmarks)

## Benchmarks
| Model         | Auto-Batch Memory Usage | Auto-Batch Seconds per Image | 1 Batch Memory Usage | 1 Batch Seconds per Image |
|--------------|------------------------|-----------------------------|----------------------|-------------------------|
| LLaVA 7B     | 17,978 MB               | 3.25                        | 7,014 MB             | 3.62                    |
| ViT-GPT2 0.27B | 7,570 MB                | 0.08                        | 914 MB               | 0.79                    |
| BLIPv2 2.7B  | 13,534 MB               | 0.25                        | 4,590 MB             | 2.53                    |

All models were tested on 502 random image from kaggle dataset found [here](https://www.kaggle.com/datasets/cyanex1702/cyberversecyberpunk-imagesdataset). Images were resized based on their config files and tested on GeForce RTX 4090 Graphics Card with 24 Gb memory.

## Installation
GenIText is available as a Python package and can be installed easily using `pip`. 

To install GenIText, simply run:
```bash
pip install genitext
```
After installation, you can verify that the CLI tool is accessible by running:
```bash 
genitext --help
```
To initiate the CLI tool, run: 
```bash
genitext
```
GenIText incorporates LLMs from Ollama to assist with prompt refinement which means ollama has to be available on the device when running `/refine` in the CLI tool. You can download the software for Mac or Windows OS from [here](https://ollama.com/download/). For Linux OS, you can install directly via the following: 
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
After installing, pull the appropriate LLM you want to use in `/refine`. Currently, the default config is set to `deepseek-r1:7b` since it offers strong performance with it's reasoning capabilities while using relatively trivial memory. Options to switch the LLM in config will be made available soon.