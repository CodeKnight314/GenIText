# GenITA: Generative Image-Text Automated package

<p align="center">
  <img src="resources/gif/demo.gif" alt="Demonstration video of GenITA tool">
</p>

## Overview
This repository is independently developed as a felxible framework to generate high-quality Image-Text pairs for finetuning Image-Generation models, such as Stable Diffusion, DALL-E, and other generative models. By leveraging open-source captioning models, GenITA automates the process of generating diverse captions for corresponding images, ensuring that the text data is well-suited for downstream applications such as style-specific generations or domain adaptation. This framework is designed to complement contemporary repositories or modules in the field, offering an additional option for flexibility and automation to create customized datasets.

GenITA will become distributable as a CLI tool once package is ready for testing across systems. Please support in any way you see fit!

## Table of Contents
- [Installation](#installation)
- [Results](#results)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
  - [Dataset Preparation](#dataset-preparation)

## Installation
Soon to be released as python package for direct install on terminal!

1. Clone the repository:
    ```bash
    git clone https://github.com/CodeKnight314/Image-Captioning.git
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv genita-env
    source genita-env/bin/activate
    ```

3. cd to project directory: 
    ```bash 
    cd GenITA/
    ```

4. Install the required packages:
    ```bash
    pip install -r GenITA/requirements.txt
    ```
