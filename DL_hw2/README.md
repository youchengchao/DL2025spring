# Multi-Task Vision with MobileNetV3-Small

![Python](https://img.shields.io/badge/python-3.7%2B-blue) ![PyTorch](https://img.shields.io/badge/pytorch-1.13%2B-red) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

This repository provides end-to-end implementations for three computer vision tasks—Semantic Segmentation, Object Detection, and Image Classification—using MobileNetV3-Small as a shared backbone for efficiency and speed.

---

## 1. Tasks Overview

* **Semantic Segmentation**: Pixel-level labeling (VOC-style dataset).
* **Object Detection**: Bounding box prediction (COCO-style dataset).
* **Image Classification**: Category classification (Imagenette-style dataset).

Each task has its own training and evaluation script, but all share the MobileNetV3-Small feature extractor and a unified configuration system.

---

## 2. Dataset Preparation

* **VOC (Segmentation)**:

  * Layout: `VOCdevkit/VOC2012/{JPEGImages,SegmentationClass,ImageSets/Segmentation}`
  * Use `scripts/build_mini_voc_seg.py` to create a smaller subset.

* **COCO (Detection)**:

  * Layout: `images/` and `annotations/` directories.
  * Use `scripts/build_mini_coco.py` for subset creation.

* **Imagenette (Classification)**:

  * Layout: `train/<class>/`, `val/<class>/`.
  * Download from [Imagenette160](https://github.com/fastai/imagenette).

Custom datasets can be defined by updating paths in `configs/*.yaml`.

---

## 3. Model Architecture

* **Backbone**: MobileNetV3-Small (depthwise separable conv + squeeze-and-excitation blocks).
* **Segmentation Head**: Atrous Spatial Pyramid Pooling (ASPP) module + decoder with skip-connections.
* **Detection Head**: SSD-style anchor predictions (also supports Faster R-CNN).
* **Classification Head**: Global average pooling + dropout + fully connected layer.

Optimizer: **LBFGS** with **CosineAnnealingWarmRestarts** scheduler for all tasks.

---

## 4. Environment & Installation

### 4.1 Hardware Recommendations

* **GPU**: NVIDIA GeForce GTX 1660 Ti (6GB VRAM or higher)
* **CPU**: Intel(R) Core(TM) i5-9400F @ 2.90GHz (6 cores)
* **Memory**: 16 GB RAM

### 4.2 Generate Requirement Files

```bash
conda activate your_env_name
conda list --export > conda-requirements.txt
pip freeze > pip-requirements.txt
```

### 4.3 Create Environment & Install

```bash
# 1. Create conda environment from conda list
conda create -n vision-env --file conda-requirements.txt
conda activate vision-env

# 2. Install pip packages
pip install -r pip-requirements.txt
```

---

## 5. Usage & Commands

Each task has a dedicated script. Common arguments:

|            Arg | Description                                   | Default              |
| -------------: | --------------------------------------------- | -------------------- |
|       `--task` | `segmentation`, `detection`, `classification` | `detection`          |
|   `--data-dir` | Path to dataset root                          | (required)           |
|   `--backbone` | Backbone name                                 | `mobilenet_v3_small` |
|     `--epochs` | Number of epochs                              | `50`                 |
| `--batch-size` | Training batch size                           | `16`                 |
|    `--init_lr` | Initial learning rate                         | `1e-3`               |
|  `--n_workers` | DataLoader workers                            | `4`                  |
| `--output-dir` | Directory for outputs                         | `outputs/`           |

### 5.1 Segmentation

```bash
python train_segmentation.py \
  --task voc \
  --data-dir /path/to/VOCdevkit/VOC2012 \
  --epochs 80 --batch-size 8 --output-dir outputs/seg
```

### 5.2 Detection

```bash
python train_detection.py \
  --task coco \
  --data-dir /path/to/coco \
  --epochs 50 --batch-size 16 --output-dir outputs/det
```

### 5.3 Classification

```bash
python train_classification.py \
  --task imagenette \
  --data-dir /path/to/imagenette160 \
  --epochs 30 --batch-size 128 --output-dir outputs/cls
```

---

## 6. Configuration

All hyperparameters and augmentations live in YAML under `configs/`:

* `configs/segmentation.yaml`
* `configs/detection.yaml`
* `configs/classification.yaml`

Modify learning rates, scheduler settings, input size, augmentations, and dataset paths.

---

## 7. Project Structure

```
your-project/
├── configs/                # YAML configs
├── data/                   # Raw & mini datasets
├── models/                 # Backbone and heads
├── scripts/                # Dataset builders, utils
├── train_segmentation.py
├── train_detection.py
├── train_classification.py
├── utils/                  # DataLoaders, transforms, metrics, viz
├── weights/                # Pretrained MobileNetV3-Small
├── outputs/                # Checkpoints and logs
├── conda-requirements.txt
├── pip-requirements.txt
└── README.md
```

---

## 8. Contributing

1. Fork the repo
2. Create branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add feature"`
4. Push and open PR

Please follow the [Contributor Covenant](CODE_OF_CONDUCT.md).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
