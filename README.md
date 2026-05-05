## AI Usage and External Sources
We used AI for non-core support tasks, including finding documentation/examples, README formatting, and understanding results for learning purposes. AI was not used for the core logic or initial project approach. The external sources below were used as references for documentation, code examples, and similar transfer learning/image classification workflows.
* [https://archive.ics.uci.edu/dataset/908/realwaste](https://archive.ics.uci.edu/dataset/908/realwaste)
* [https://www.mdpi.com/2078-2489/14/12/633](https://www.mdpi.com/2078-2489/14/12/633)
* [https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
* [https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)
* [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)
* [https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)
* [https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
* [https://docs.pytorch.org/vision/main/transforms.html](https://docs.pytorch.org/vision/main/transforms.html)
* https://github.com/bentrevett/pytorch-image-classification
* [https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.CrossEntropyLoss.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.CrossEntropyLoss.html)
* [https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
* [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
* [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)
* [https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
* [https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html)
* [https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html)
* https://github.com/RobustBench/robustbench
* [https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html](https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
* [https://arxiv.org/abs/1903.12261](https://arxiv.org/abs/1903.12261)
* [https://github.com/hendrycks/robustness](https://github.com/hendrycks/robustness)
* [https://docs.pytorch.org/docs/stable/notes/randomness.html](https://docs.pytorch.org/docs/stable/notes/randomness.html)
* [https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)




# Robust Waste Material Classification Using Transfer Learning


## Team

- Haider Khan
- Bat-Amgalan 

Course: MSML640 Computer Vision  
Project Type: Image Classification with Transfer Learning

---

## 1. Project Overview

This project builds a computer vision model that classifies waste images into four recyclable material categories:

- cardboard
- glass
- metal
- plastic

The practical motivation is waste sorting. In real recycling systems, incorrect sorting of plastic, glass, cardboard, and metal can reduce recycling efficiency and increase contamination.

The deeper computer vision question is:

> Does a pretrained CNN actually learn material features such as texture, reflection, transparency, and shape, or does it rely on visual shortcuts such as color, background, lighting, object position, and image cleanliness?

To study this, we fine-tuned an ImageNet-pretrained EfficientNet-B0 model and compared four training configurations:

1. Baseline: real training images only
2. Real data + augmentation
3. Real data + synthetic data
4. Real data + synthetic data + augmentation

We also evaluated the best model under realistic perturbations such as blur, noise, brightness changes, dark lighting, and occlusion.

---

## 2. Dataset

We use a 4-class subset of the RealWaste image classification dataset (The UCI Machine Learning Repository)

Original selected class counts:

- Cardboard: 461 images
- Glass: 420 images
- Metal: 790 images
- Plastic: 921 images

For this project, we use a balanced subset:

| Split     | Images per Class | Total Images |
|-----------|------------------|--------------|
| Train     | 70               | 280          |
| Validation| 15               | 60           |
| Test      | 15               | 60           |

The raw data should be organized as:

```text
data/raw/cardboard
data/raw/glass
data/raw/metal
data/raw/plastic
```

After running the data preparation script, the project creates:

```text
data/train/
data/val/
data/test/
data/synthetic_train/
```

Synthetic images are generated only from the training split. Validation and test images remain real images only.

---

## 3. Model

We use **EfficientNet-B0 pretrained on ImageNet**.

Reason for choosing EfficientNet-B0:

- It is lightweight compared with larger CNNs.
- It is suitable for small-to-medium image classification tasks.
- ImageNet pretraining gives useful low-level and mid-level visual features such as edges, textures, shapes, and object patterns.
- It is efficient enough to train on a laptop.

The feature extractor is frozen, and the final classifier layer is replaced with a new 4-class classifier.

---

## 4. Training Configurations

1. Baseline | Real training data only 
2. Augmentation | Real training data with random crop, flip, rotation, and color jitter 
3. Synthetic | Real training data plus synthetic training images 
4. Synthetic + Augmented | Real + synthetic data with augmentation 

Synthetic images are generated using controlled perturbations such as blur, brightness changes, contrast changes, rotation, and occlusion. These simulate real-world camera conditions such as motion blur, poor lighting, partial blockage, and imperfect framing.

---

## 5. Environment Setup

This project was tested with Python 3.11.

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Important dependency note:

```text
numpy==1.26.4
```

This version is pinned because some PyTorch/torchvision installations may fail with NumPy 2.x.

---

## 6. Prepare Dataset

## Data Reproduction Note

The GitHub repository includes the prepared project data splits:

```text
data/train/
data/val/
data/test/
data/synthetic_train/

The large original raw dataset folder is not included in the repository because of size. Therefore, if you are using the submitted repository as-is, you can skip:


```bash
python src/prepare_data.py
python src/create_synthetic_data.py
```

Expected output:

```text
cardboard: 70 train, 15 val, 15 test
glass: 70 train, 15 val, 15 test
metal: 70 train, 15 val, 15 test
plastic: 70 train, 15 val, 15 test

Created 70 synthetic images for cardboard
Created 70 synthetic images for glass
Created 70 synthetic images for metal
Created 70 synthetic images for plastic
```

---

## 7. Train Models

Final reported results use:

```bash
USE_IMAGENET_NORM=0
```

Run all four training configurations:

```bash
USE_IMAGENET_NORM=0 python -m src.train --config baseline
USE_IMAGENET_NORM=0 python -m src.train --config augmentation
USE_IMAGENET_NORM=0 python -m src.train --config synthetic
USE_IMAGENET_NORM=0 python -m src.train --config synthetic_augmented
```

Model checkpoints are saved in:

```text
results/models/
```

Loss and accuracy curves are saved in:

```text
results/loss_curves/
```

---

## 8. Evaluate Models

Run:

```bash
USE_IMAGENET_NORM=0 python -m src.evaluate --config baseline
USE_IMAGENET_NORM=0 python -m src.evaluate --config augmentation
USE_IMAGENET_NORM=0 python -m src.evaluate --config synthetic
USE_IMAGENET_NORM=0 python -m src.evaluate --config synthetic_augmented
```

Confusion matrices are saved in:

```text
results/confusion_matrices/
```

---

## 9. Final Test Results (Configuration & Test Accuracy)

1. Synthetic **73.33%** 
2. Augmentation 71.67%
3. Baseline 70.00%
4. Synthetic + Augmented 66.67% 


The synthetic-data model improves over the baseline from 70.00% to 73.33% making the syntheic model as best model.

---

## 10. Robustness Testing

The best model was evaluated under realistic image perturbations.

Run:

```bash
USE_IMAGENET_NORM=0 python -m src.robustness_eval --config synthetic
```

Robustness results are saved in:

```text
results/robustness/synthetic_robustness_results.csv
```

Robustness results: (Perturbation & Accuracy)

| Clean | 73.33% |
| Blur | 70.00% |
| Noise | 60.00% |
| Dark lighting | 61.67% |
| Bright lighting | 68.33% |
| Occlusion | 68.33% |

Main observation: the model remains fairly stable under blur, bright lighting, and occlusion, but performance drops more under noise and dark lighting.

---

## 11. Error Examples

Misclassified examples are saved using:

```bash
USE_IMAGENET_NORM=0 python -m src.collect_error_examples --config synthetic --max_examples 20
```

Outputs are saved in:

```text
results/error_examples/synthetic/
```

The error examples show that the model often confuses glass, metal, and plastic. This makes sense because these materials can share visual properties such as shine, transparency, reflection, and irregular packaging shape.

---

## 12. Preprocessing Ablation

We also compared performance with and without ImageNet normalization.

| Preprocessing | Baseline | Augmentation | Synthetic | Synthetic + Augmented |
|---|---:|---:|---:|---:|
| No normalization | 70.00% | 71.67% | **73.33%** | 66.67% |
| ImageNet normalization | 61.67% | 66.67% | 70.00% | 70.00% |

Although ImageNet normalization is commonly used with pretrained models, the no-normalization setting performed better in this frozen-backbone experiment. Therefore, the final reported results use `USE_IMAGENET_NORM=0`.

---

## 13. Repository Structure

```text
.
├── data
│   ├── raw
│   ├── synthetic_train
│   ├── test
│   ├── train
│   └── val
├── notebooks
├── report
├── results
│   ├── confusion_matrices
│   ├── error_examples
│   ├── logs
│   ├── loss_curves
│   ├── models
│   └── robustness
├── src
│   ├── collect_error_examples.py
│   ├── config.py
│   ├── create_synthetic_data.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── prepare_data.py
│   ├── robustness_eval.py
│   └── train.py
├── README.md
└── requirements.txt
```

---

## 15. How to Reproduce Final Results

From a clean run:

```bash
source .venv/bin/activate

rm -rf results
mkdir -p results/logs

python src/prepare_data.py | tee results/logs/prepare_data.log
python src/create_synthetic_data.py | tee results/logs/create_synthetic_data.log

USE_IMAGENET_NORM=0 python -m src.train --config baseline | tee results/logs/train_baseline.log
USE_IMAGENET_NORM=0 python -m src.train --config augmentation | tee results/logs/train_augmentation.log
USE_IMAGENET_NORM=0 python -m src.train --config synthetic | tee results/logs/train_synthetic.log
USE_IMAGENET_NORM=0 python -m src.train --config synthetic_augmented | tee results/logs/train_synthetic_augmented.log

USE_IMAGENET_NORM=0 python -m src.evaluate --config baseline | tee results/logs/eval_baseline.log
USE_IMAGENET_NORM=0 python -m src.evaluate --config augmentation | tee results/logs/eval_augmentation.log
USE_IMAGENET_NORM=0 python -m src.evaluate --config synthetic | tee results/logs/eval_synthetic.log
USE_IMAGENET_NORM=0 python -m src.evaluate --config synthetic_augmented | tee results/logs/eval_synthetic_augmented.log

USE_IMAGENET_NORM=0 python -m src.robustness_eval --config synthetic | tee results/logs/robustness_eval_synthetic.log
USE_IMAGENET_NORM=0 python -m src.collect_error_examples --config synthetic --max_examples 20 | tee results/logs/error_examples_synthetic.log
```

---

## 16. Main Conclusion

Synthetic data gave the best test accuracy, improving performance from 70.00% for the baseline to 73.33%. This suggests that controlled synthetic perturbations can help generalization. However, the improvement is modest, and the model still struggles with visually similar materials such as glass, plastic, and metal. The project shows that the model learns useful visual patterns, but it may still rely on shortcuts such as shine, color, transparency, and object shape rather than true material identity.
