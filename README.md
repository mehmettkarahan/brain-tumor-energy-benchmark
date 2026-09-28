# Brain Tumor MRI CNN Energy Benchmark

This repository provides a model-by-model organization of the experiment code and outputs for four-class brain-tumor MRI classification and CPU/GPU training-energy analysis.

## Dataset

Public dataset: **Augmented Brain Tumor MRI Image Dataset** (Kaggle)  
https://www.kaggle.com/datasets/hghdhygf/brain-tumor-mri-image-dataset

The dataset itself is not redistributed in this repository. The experiments use ImageFolder-compatible training and test directories containing four classes: glioma, meningioma, notumor, and pituitary.

## Core training settings

- 25 epochs
- Batch size: 32
- Adam optimizer
- Learning rate: 1e-4
- CrossEntropyLoss
- Models trained from scratch (no pretrained weights)
- StepLR(step_size=7, gamma=0.1) in the cleaned model scripts
- ImageNet normalization

For the main 224 x 224 protocol, images are resized to 224 x 224, converted to tensors, and normalized using the ImageNet mean and standard deviation.

## Energy calculation

`energy_summary.csv` contains the model-level values used in the manuscript table, including DenseNet121. Energy is calculated from average power and elapsed time:

`E = P_avg x t`

where power is measured in watts (W) and time in seconds (s). Energy in kilojoules is calculated as:

`E_kJ = (P_avg x t) / 1000`

CPU and GPU energy are calculated separately and total energy is their sum:

`E_total = E_CPU + E_GPU`

Each model folder contains a reduced `power_log.csv` with the extracted HWiNFO timestamp, CPU all-power, and GPU-power samples used for compact repository documentation. The complete HWiNFO exports are not redistributed because they contain many unrelated hardware sensors and are substantially larger.

## Repository layout

```text
models/
  AlexNet/
  DarkNet19/
  DenseNet121/
  EfficientNetB0/
  GoogLeNet/
  MobileNetV2/
  ResNet18/
  ResNet34/
  ResNet50/
  VGG16/
energy_summary.csv
power_log_extracted_summary.csv
requirements.txt
```

Each model folder contains a standalone `train.py`, a short model README, and the available result files. The combined development scripts, duplicate console outputs, and full multi-sensor HWiNFO CSV exports are intentionally omitted from this GitHub-oriented package.

## Evaluation protocol note

The provided model scripts evaluate the held-out test directory after each epoch and perform a final evaluation after training. A stricter future experimental design can introduce a separate validation partition and reserve the test partition for final evaluation only.
