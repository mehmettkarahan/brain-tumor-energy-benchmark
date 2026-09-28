# Brain Tumor MRI CNN Energy Benchmark

This repository contains the model-specific training code, evaluation outputs, and energy-consumption summaries used in a four-class brain-tumor MRI classification benchmark. The study compares ten convolutional neural network (CNN) architectures in terms of classification performance, training time, and CPU/GPU energy consumption under a consistent experimental environment.

## Dataset

The experiments use the **Augmented Brain Tumor MRI Image Dataset** available on Kaggle:

https://www.kaggle.com/datasets/hghdhygf/brain-tumor-mri-image-dataset

The dataset itself is **not redistributed** in this repository. Training and test data are expected to follow a PyTorch `ImageFolder`-compatible directory structure with four classes:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

Example structure:

```text
train/
  glioma/
  meningioma/
  notumor/
  pituitary/

test/
  glioma/
  meningioma/
  notumor/
  pituitary/
```

## Evaluated Models

The benchmark includes the following CNN architectures:

- AlexNet
- DarkNet19
- DenseNet121
- EfficientNetB0
- GoogLeNet
- MobileNetV2
- ResNet18
- ResNet34
- ResNet50
- VGG16

All models are trained **from scratch**, without pretrained weights.

## Core Training Configuration

| Parameter | Configuration |
|---|---|
| Number of classes | 4 |
| Epochs | 25 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | `1e-4` |
| Loss function | CrossEntropyLoss |
| Learning-rate scheduler | StepLR (`step_size=7`, `gamma=0.1`) |
| Pretrained weights | No |
| Final input size | 224 × 224 |
| Normalization | ImageNet mean and standard deviation |

Most model scripts directly resize images to `224 × 224`. The `EfficientNetB0` and `ResNet34` scripts use `Resize(256)` followed by `CenterCrop(224)`. All scripts then convert images to tensors and apply ImageNet normalization:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

## System and Tool Configuration

All experiments were conducted using the hardware and software environment summarized below. Using the same experimental platform for all architectures helps reduce system-level variability when comparing training time and measured energy consumption.

| Category | Component | Configuration |
|---|---|---|
| **Hardware** | System architecture | x64-based computer |
| **Hardware** | CPU | AMD Ryzen 7 4800H @ 2.90 GHz, 8 cores / 16 threads, 45 W TDP |
| **Hardware** | GPU | NVIDIA GeForce GTX 1650 Ti, 4 GB GDDR6, 50 W TDP |
| **Hardware** | RAM | 16 GB DDR4-3200 MHz (2 × 8 GB) |
| **Software** | Operating system | Microsoft Windows 11 Pro |
| **Software** | Python | 3.10.11 |
| **Software** | PyTorch | 2.6.0+cu118 |
| **Software** | TorchVision | 0.21.0+cu118 |
| **Software** | CUDA | 11.8 |
| **Energy Measurement** | Monitoring tool | HWiNFO64 8.24-5700 |

## Energy Measurement and Calculation

CPU and GPU power measurements were obtained using **HWiNFO64 8.24-5700**. Energy is calculated from average power and elapsed training time:

```text
E = P_avg × t
```

where:

- `E` is energy in joules (J),
- `P_avg` is average power in watts (W),
- `t` is elapsed time in seconds (s).

Energy in kilojoules is calculated as:

```text
E_kJ = (P_avg × t) / 1000
```

CPU and GPU energy are calculated separately:

```text
E_CPU = (P_CPU,avg × t) / 1000
E_GPU = (P_GPU,avg × t) / 1000
```

Total training energy is then:

```text
E_total = E_CPU + E_GPU
```

The file [`energy_summary.csv`](energy_summary.csv) contains the model-level values used for the benchmark summary. Each model folder also contains a reduced `power_log.csv` with the extracted HWiNFO timestamp, CPU all-power, and GPU-power samples used for compact repository documentation.

The complete multi-sensor HWiNFO exports are not redistributed because they contain many unrelated hardware sensor channels and are substantially larger than the reduced logs included here.

## Benchmark Results

| Model | Accuracy (%) | Elapsed Time (s) | CPU Avg. Power (W) | GPU Avg. Power (W) | Total Energy (kJ) |
|---|---:|---:|---:|---:|---:|
| AlexNet | 93.62 | 1,909 | 29.19 | 38.88 | 129.94 |
| ResNet18 | 94.88 | 3,279 | 29.78 | 42.88 | 238.25 |
| MobileNetV2 | 88.38 | 3,663 | 28.84 | 43.20 | 263.91 |
| GoogLeNet | 98.38 | 3,932 | 28.93 | 44.04 | 286.94 |
| EfficientNetB0 | 93.50 | 4,547 | 27.74 | 44.10 | 326.65 |
| DarkNet19 | 97.12 | 4,580 | 27.18 | 43.84 | 325.25 |
| ResNet34 | 96.00 | 4,909 | 26.70 | 44.44 | 349.22 |
| ResNet50 | 95.38 | 8,564 | 20.53 | 46.26 | 571.93 |
| DenseNet121 | 99.50 | 54,666 | 10.05 | 35.97 | 2,515.84 |
| VGG16 | 94.12 | 182,096 | 11.37 | 35.96 | 8,618.69 |

The complete numerical summary, including separate CPU and GPU energy values, is available in [`energy_summary.csv`](energy_summary.csv). The extracted power-log means are summarized in [`power_log_extracted_summary.csv`](power_log_extracted_summary.csv).

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── energy_summary.csv
├── power_log_extracted_summary.csv
└── models/
    ├── AlexNet/
    ├── DarkNet19/
    ├── DenseNet121/
    ├── EfficientNetB0/
    ├── GoogLeNet/
    ├── MobileNetV2/
    ├── ResNet18/
    ├── ResNet34/
    ├── ResNet50/
    └── VGG16/
```

Each model directory contains:

```text
ModelName/
├── README.md
├── train.py
└── results/
    ├── confusion_matrix.png
    ├── power_log.csv
    ├── roc_curve.png
    ├── training_metrics.png
    └── training_results.txt
```

The combined development scripts, duplicate console outputs, and full multi-sensor HWiNFO exports are intentionally omitted from this GitHub-oriented package.

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/mehmettkarahan/brain-tumor-energy-benchmark.git
cd brain-tumor-energy-benchmark
pip install -r requirements.txt
```

The exact software versions used in the reported experiments are listed in the **System and Tool Configuration** section above. The `requirements.txt` file lists the core package dependencies required by the training scripts.

## Running a Model

Each architecture has a standalone `train.py` script. For example, to run AlexNet:

```bash
python models/AlexNet/train.py \
  --train-dir /path/to/train \
  --test-dir /path/to/test \
  --output-dir results_new
```

On Windows PowerShell, the same command can be entered on one line:

```powershell
python models/AlexNet/train.py --train-dir "C:\path\to\train" --test-dir "C:\path\to\test" --output-dir results_new
```

Replace `AlexNet` with another model directory name to run a different architecture.

## Generated Outputs

A training run generates evaluation artifacts such as:

- training/evaluation loss and accuracy curves,
- confusion matrix,
- ROC curve,
- textual training and evaluation results.

The repository also includes the result files associated with the reported benchmark for each model.

## Evaluation Protocol Note

The provided model scripts evaluate the held-out test directory after each epoch and perform a final evaluation after training. Therefore, the repository reflects the evaluation protocol used by these cleaned scripts.

For future experiments requiring stricter separation of model development and final performance estimation, a dedicated validation split can be introduced for epoch-level model assessment, while the test set is reserved for the final evaluation only.

## Reproducibility Notes

- The same hardware platform was used across the reported model runs.
- Models were trained from scratch with no pretrained weights.
- The dataset is not included in this repository and must be obtained separately from the source linked above.
- Reduced power logs are included for repository compactness; full HWiNFO multi-sensor exports are not included.
- Exact runtime and energy measurements can vary with hardware, operating conditions, background processes, driver versions, and software configuration.

## Citation / Associated Study

If you use this repository, please cite the associated article once its bibliographic information is available. The article citation or DOI can be added here after publication.
