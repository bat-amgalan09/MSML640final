# Robust Waste Material Classification Using Transfer Learning

## Task

This project classifies waste images into four recyclable material categories:

- cardboard
- glass
- metal
- plastic

## Dataset

We use a 4-class subset of the RealWaste dataset. The raw data should be organized as:

```text
data/raw/cardboard
data/raw/glass
data/raw/metal
data/raw/plastic


## How to Run

This project was tested with Python 3.11.

### 1. Create environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


2. Prepare data split

The raw dataset should be organized as:

data/raw/cardboard
data/raw/glass
data/raw/metal
data/raw/plastic

Run:
python src/prepare_data.py
python src/create_synthetic_data.py


3. Train all four configurations
python -m src.train --config baseline
python -m src.train --config augmentation
python -m src.train --config synthetic
python -m src.train --config synthetic_augmented



4. Evaluate all four configurations

python -m src.evaluate --config baseline
python -m src.evaluate --config augmentation
python -m src.evaluate --config synthetic
python -m src.evaluate --config synthetic_augmented

Outputs are saved in:
results/models/
results/loss_curves/
results/confusion_matrices/



## Qualitative Error Analysis

To better understand model failures, we saved misclassified test images from the best test model. The most common mistakes occurred between glass, plastic, and metal. These classes share visual properties such as reflection, shine, transparency, and irregular shape.

Cardboard was the easiest class because it often has a consistent brown color and texture. In contrast, glass was sometimes predicted as plastic or metal when the object was transparent, reflective, or partially occluded. Metal and plastic were also confused when plastic packaging had shiny surfaces.

These errors suggest that the model is not always learning true material identity. Instead, it may rely partly on visual shortcuts such as brightness, reflectiveness, color, object shape, and background context.


### Raw DataSet
The dataset used in this project is a 4-class subset of RealWaste:

- Cardboard: 461 images
- Glass: 420 images
- Metal: 790 images
- Plastic: 921 images

We removed other classes such as Paper, Food Organics, Vegetation, Textile Trash, and Miscellaneous Trash to focus on common recyclable materials and keep the project aligned with our proposal.