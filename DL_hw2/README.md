# Object Detection, Semantic Segmentation & Image Classification

This project focuses on Object Detection, Semantic Segmentation, and Image Classification. Pretrained weights of MobileNet-V3 Small are used as the initial weights.

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Architecture](#model-architecture)  
- [Pretrained Weights](#pretrained-weights)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview

This repository implements three tasks on images:

1. **Object Detection**  
2. **Semantic Segmentation**  
3. **Image Classification**  

All models are initialized with pretrained weights from MobileNet-V3 Small to accelerate convergence and improve accuracy.

---

## Features

- Train and evaluate an object detection, semantic segmentation and image classification model using MobileNet-V3 Small as the backbone.
- Custom datasets: mini-VOC, mini-COCO, imagenette160.  
- Configuration files for switching between tasks and datasets easily. 
- Use LBFGS optimizer cooperating with CosineAnnealingWarmRestarts scheduler

---

## Prerequisites

- Python 3.7 or higher  
- PyTorch 1.13+ (CUDA 10.2/11.1 or CPU only)  
- torchvision 0.14+  
- fastai 2.7+ (optional, if using fastai data pipelines)  
- OpenCV 4.x  
- tqdm, numpy, pillow, matplotlib  

---

## Installation

1. Clone this repository:
  ```bash
  cd your-project
  python -m venv venv
  pip install -r requirements.txt
  ```

2. Create a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate       # Windows
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** Adjust the versions of PyTorch/torchvision according to your CUDA version:
   >
   > ```bash
   > pip install torch==1.13.1 torchvision==0.14.1
   > ```

---

## Usage

### 1. Object Detection

```bash
python train_detection.py \
  --data-dir /path/to/dataset \
  --task coco \
  --backbone mobilenet_v3_small \
  --epochs 50 \
  --batch-size 16 \
  --output-dir outputs/detection
```

* `--data-dir` points to the dataset root (contains `images/` and `annotations/`).
* `--task` can be `coco`, `voc`, or `custom`.
* `--backbone` defaults to `mobilenet_v3_small`.
* Check `configs/detection.yaml` for additional hyperparameters (learning rate, scheduler, etc.).

### 2. Semantic Segmentation

```bash
python train_segmentation.py \
  --data-dir /path/to/voc2012 \
  --task voc \
  --backbone mobilenet_v3_small \
  --epochs 80 \
  --batch-size 8 \
  --output-dir outputs/segmentation
```

* VOC datasets expect the standard folder layout:

  ```
  VOCdevkit/
  └── VOC2012/
      ├── JPEGImages/
      ├── SegmentationClass/
      └── ImageSets/Segmentation/{train.txt,val.txt}
  ```

* Check `configs/segmentation.yaml` for details on input size, loss weights, and augmentation.

### 3. Image Classification

```bash
python train_classification.py \
  --data-dir /path/to/imagenet \
  --backbone mobilenet_v3_small \
  --epochs 30 \
  --batch-size 128 \
  --output-dir outputs/classification
```

* For ImageNet or Imagenette, ensure your data is organized as:

  ```
  ImageNet/
  ├── train/
  │   ├── class1/
  │   ├── class2/
  │   └── ...
  └── val/
      ├── class1/
      ├── class2/
      └── ...
  ```
* Hyperparameters and augmentations are defined in `configs/classification.yaml`.

---

## Model Architecture

* **Backbone**
  All tasks share MobileNet-V3 Small as the feature extractor.

  * Depthwise‐separable convolutions with squeeze-and-excitation.
  * Lightweight and optimized for mobile/embedded inference.

* **Object Detection Head**

  * Single-stage (SSD) or Two-stage (Faster R-CNN) heads supported.
  * Default anchors and feature map sizes configured for COCO.

* **Semantic Segmentation Head**

  * Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context.
  * Decoder with upsampling and skip-connections (similar to DeepLabV3+).

* **Classification Head**

  * Global average pooling followed by a fully connected layer.
  * Dropout before the final classification layer.

---

## Pretrained Weights

* We provide pretrained MobileNet-V3 Small weights trained on ImageNet.

* If you want to use your own pretrained `.pth` file, place it under `weights/` and update the configuration:

  ```yaml
  model:
    backbone: mobilenet_v3_small
    pretrained: False
    weights_path: weights/custom_mobilenetv3_small.pth
  ```

* Default training scripts will load `weights/mobilenet_v3_small_pretrained.pth` if `pretrained: True`.

---

## Project Structure

```
your-project/
├── configs/
│   ├── classification.yaml
│   ├── detection.yaml
│   └── segmentation.yaml
├── data/                      # Where raw datasets and mini‐datasets reside
│   ├── VOCdevkit/
│   ├── COCO/
│   └── custom/
├── models/                    # Model definitions
│   ├── backbones/
│   │   └── mobilenet_v3_small.py
│   ├── detection_head.py
│   ├── segmentation_head.py
│   └── classification_head.py
├── scripts/                   # Utilities for building mini‐datasets, etc.
│   └── build_mini_voc_seg.py
├── train_classification.py
├── train_detection.py
├── train_segmentation.py
├── utils/
│   ├── dataset_loaders.py
│   ├── transforms.py
│   ├── metrics.py
│   └── visualization.py
├── weights/                   # Pretrained weights (mobilenet_v3_small_pretrained.pth)
├── outputs/                   # Saved checkpoints and logs
├── requirements.txt
└── README.md                  # ← You are here
```

---

## Contributing

1. Fork this repository.
2. Create a feature branch (e.g., `git checkout -b feature/awesome‐feature`).
3. Commit your changes (`git commit -m "Add awesome feature"`).
4. Push to the branch (`git push origin feature/awesome‐feature`).
5. Open a Pull Request.

Please follow the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.