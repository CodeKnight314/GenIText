# Generative Image-Text Automated package
## Overview
This repository is independently developed as a felxible framework to generate high-quality Image-Text pairs for finetuning Image-Generation models, such as Stable Diffusion, DALL-E, and other generative models. By leveraging open-source captioning models, GITA automates the process of generating diverse captions for corresponding images, ensuring that the text data is well-suited for downstream applications such as style-specific generations or domain adaptation. This framework is designed to complement, not replace, contemporary repositories or modules in the field, offering an additional option for flexibility and automation to create customized datasets.

## Table of Contents
- [Installation](#installation)
- [Results](#results)
- [Usage](#usage)
  - [Inference](#inference)
  - [Training](#training)
  - [Dataset Preparation](#dataset-preparation)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/CodeKnight314/Image-Captioning.git
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv gita-env
    source gita-env/bin/activate
    ```

3. cd to project directory: 
    ```bash 
    cd GITA/
    ```

4. Install the required packages:
    ```bash
    pip install -r GITA/requirements.txt
    ```
