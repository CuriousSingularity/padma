<p align="center">
  <img src="docs/icon.svg" alt="Padma Logo" width="200"/>
</p>

# Padma

**🌸 A modular PyTorch training framework for quick experimentation with any model architecture and any dataset.**

Padma lets you rapidly prototype and experiment with different combinations of models and data without writing boilerplate code. Simply define your data and model architecture through configuration files, and start training immediately. Built on PyTorch Lightning for robust training, Hydra for flexible configuration management, and timm for access to hundreds of pretrained models.

## ✨ Key Features

- **🎯 Any Model Architecture**: Instant access to 100+ pretrained models from timm (ResNets, EfficientNets, Vision Transformers, ConvNeXt, MobileNets, and more) or easily plug in your own custom architectures
- **📊 Any Dataset**: Built-in support for MNIST, CIFAR, ImageNet, or bring your own data with minimal setup
- **⚡ Rapid Experimentation**: Switch models, datasets, and hyperparameters with simple command-line overrides - no code changes needed
- **⚙️ Configuration-Driven**: All experiments defined through composable YAML configs using Hydra
- **🚀 Production-Ready Training**: Built on PyTorch Lightning with automatic mixed precision, gradient accumulation, checkpointing, early stopping, and comprehensive logging
- **🧩 Modular & Extensible**: Clean architecture makes it easy to add new models, datasets, or training strategies

## 🤔 Why Padma?

**Stop writing boilerplate. Start experimenting.**

Traditional deep learning workflows require writing custom training loops, dataset classes, and configuration code before you can even begin experimentation. Padma eliminates this friction:

- **⏱️ Test ideas in minutes, not hours**: No need to write training loops or data loading code
- **🔄 Compare models effortlessly**: Try EfficientNet vs ResNet vs ViT with a single parameter change
- **💾 Validate on your data immediately**: Drop in your images, specify the folder, and train
- **🔁 Reproducible by default**: Every experiment is logged with full configuration tracking
- **📈 Scale from prototype to production**: Same code runs on CPU, GPU, or Apple Silicon with optimized training

Whether you're a researcher exploring architectures, a practitioner comparing models for deployment, or a student learning deep learning, Padma gets you from idea to results faster.

## 📦 Installation

```bash
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## 🚀 Quick Start

```bash
# Train with default config (MNIST + ResNet50)
python train.py

# Train with MobileNetV3 (fastest)
python train.py +experiment=mnist_mobilenet

# Train with ResNet18
python train.py +experiment=mnist_resnet

# Train CIFAR-10 with ResNet
python train.py +experiment=cifar10_resnet
```

## 🧪 Experimentation Made Easy

The power of Padma lies in its flexibility. Mix and match any model with any dataset using simple command-line overrides:

```bash
# Experiment 1: Try EfficientNet on CIFAR-100
python train.py model.name=efficientnet_b0 dataset.name=cifar100 model.num_classes=100

# Experiment 2: Compare different Vision Transformer sizes on MNIST
python train.py model.name=vit_tiny_patch16_224 dataset.name=mnist
python train.py model.name=vit_small_patch16_224 dataset.name=mnist
python train.py model.name=vit_base_patch16_224 dataset.name=mnist

# Experiment 3: Test lightweight models for edge deployment
python train.py model.name=mobilenetv3_small_050 training.batch_size=256
python train.py model.name=efficientnet_b0 training.batch_size=256

# Experiment 4: Your custom dataset with any pretrained model
python train.py dataset.name=custom dataset.data_dir=./my_data model.name=resnet50 model.num_classes=20

# Experiment 5: Sweep hyperparameters
python train.py training.optimizer.lr=1e-4 training.batch_size=64
python train.py training.optimizer.lr=1e-3 training.batch_size=128
```

No code changes needed - just configure and run. All experiments are automatically logged with full reproducibility.

## 🎯 Training

### Basic Training

```bash
python train.py
```

### Using Experiment Configs

```bash
# MNIST experiments
python train.py +experiment=mnist_mobilenet    # Fastest (~1M params)
python train.py +experiment=mnist_resnet       # ResNet18

# CIFAR-10 experiments
python train.py +experiment=cifar10_resnet     # ResNet50

# ImageNet experiments
python train.py +experiment=imagenet_vit       # ViT-Base
```

### Override Parameters

```bash
# Change model
python train.py model.name=resnet101

# Change training parameters
python train.py training.epochs=50 training.batch_size=64 training.optimizer.lr=1e-3

# Change dataset
python train.py dataset.name=cifar100 model.num_classes=100

# Combine overrides
python train.py model.name=efficientnet_b0 training.epochs=30 training.batch_size=128
```

### 🤖 Available Models

**Padma supports any model from the [timm library](https://github.com/huggingface/pytorch-image-models)** (100+ architectures with pretrained weights). Simply use the model name as shown in timm's documentation.

Popular architectures included:

| Architecture | Model Names | Params | Use Case |
|--------------|-------------|--------|----------|
| MobileNet | `mobilenetv3_small_050`, `mobilenetv3_small_100`, `mobilenetv3_large_100` | 1-5M | Edge devices, mobile |
| ResNet | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` | 11-60M | General purpose, fast |
| EfficientNet | `efficientnet_b0` to `efficientnet_b7`, `efficientnetv2_s/m/l` | 5-66M | Best efficiency |
| ConvNeXt | `convnext_tiny`, `convnext_small`, `convnext_base` | 28-89M | Modern CNN architecture |
| ViT | `vit_tiny_patch16_224`, `vit_small_patch16_224`, `vit_base_patch16_224` | 6-86M | Transformer-based |
| Swin Transformer | `swin_tiny_patch4_window7_224`, `swin_small_patch4_window7_224` | 28-50M | Hierarchical vision transformers |
| DeiT | `deit_tiny_patch16_224`, `deit_small_patch16_224` | 5-22M | Distilled transformers |

To see all available models:
```python
import timm
print(timm.list_models(pretrained=True))
```

Or explore at [timm documentation](https://huggingface.co/docs/timm/index).

### 📁 Available Datasets

**Built-in datasets** ready to use out of the box:

| Dataset | Name | Classes | Description |
|---------|------|---------|-------------|
| MNIST | `mnist` | 10 | Handwritten digits (28x28 grayscale) |
| CIFAR-10 | `cifar10` | 10 | Natural images (32x32 RGB) |
| CIFAR-100 | `cifar100` | 100 | Natural images (32x32 RGB, fine-grained) |
| ImageNet | `imagenet` | 1000 | Large-scale natural images (requires download) |
| **Custom** | `custom` | Any | **Your own data** - just organize in folders |

**Bring your own data**: The `custom` dataset type accepts any image folder structure, making it trivial to experiment with your own datasets. See [Custom Dataset](#custom-dataset) section below for setup.

## 📊 Evaluation

After training, evaluate the model:

```bash
# Evaluate best checkpoint (Lightning saves with .ckpt extension)
python evaluate.py checkpoint_path=outputs/<date>/<time>/checkpoints/best-epoch=XX-val_accuracy=X.XXXX.ckpt

# Evaluate on different dataset
python evaluate.py checkpoint_path=path/to/checkpoint.ckpt dataset.name=cifar10
```

Evaluation outputs:
- Validation and test set metrics (accuracy, precision, recall, F1)
- Per-class accuracy breakdown
- Results saved to `evaluation_results.json`

## 📈 Monitoring

### TensorBoard

```bash
# View training logs
tensorboard --logdir=outputs/

# View specific run
tensorboard --logdir=outputs/2025-12-23/14-03-59/tensorboard
```

Logged metrics:
- Training/validation loss
- Training/validation accuracy, precision, recall, F1
- Learning rate
- Test metrics (after training)

## ⚙️ Configuration

### Directory Structure

```
configs/
├── config.yaml           # Main config
├── model/
│   └── default.yaml      # Model settings
├── dataset/
│   └── default.yaml      # Dataset settings
├── training/
│   └── default.yaml      # Training settings
└── experiment/
    ├── mnist_mobilenet.yaml
    ├── mnist_resnet.yaml
    ├── cifar10_resnet.yaml
    └── imagenet_vit.yaml
```

### Key Configuration Options

**Device** (auto-detects: CUDA → MPS → CPU):
```bash
python train.py device=auto   # Auto-detect (default)
python train.py device=cuda   # Force GPU
python train.py device=mps    # Force Apple Silicon
python train.py device=cpu    # Force CPU
```

**Mixed Precision**:
```bash
python train.py mixed_precision=true   # Enable AMP (default)
python train.py mixed_precision=false  # Disable AMP
```

**Checkpointing**:
```bash
python train.py checkpoint.save_best=true checkpoint.monitor_metric=val_accuracy
```

**Early Stopping**:
```bash
python train.py training.early_stopping.enabled=true training.early_stopping.patience=10
```

## 📂 Project Structure

```
padma/
├── models/
│   ├── base.py          # Utilities (freeze, load, info)
│   ├── resnet.py        # ResNet family
│   ├── vit.py           # Vision Transformers
│   ├── efficientnet.py  # EfficientNet family
│   ├── convnext.py      # ConvNeXt family
│   └── mobilenet.py     # MobileNet/LCNet (lightweight)
├── datasets/
│   ├── base.py          # Transforms, dataloaders
│   ├── mnist.py         # MNIST dataset
│   ├── cifar.py         # CIFAR-10/100
│   ├── imagenet.py      # ImageNet
│   └── custom.py        # Custom ImageFolder
├── trainers/
│   └── trainer.py       # Training loop
└── utils/
    ├── device.py        # Device detection
    └── metrics.py       # Metrics tracking
```

## 💼 Custom Dataset

**Using your own data is as simple as organizing images into folders.** Padma automatically handles the rest.

### Setup

Organize your images in this structure (standard ImageFolder format):

```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── img1.jpg
│   │   └── ...
│   └── class3/
│       └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── class3/
└── test/  (optional)
    ├── class1/
    ├── class2/
    └── class3/
```

### Train with Your Data

```bash
# Basic usage - Padma auto-detects number of classes
python train.py dataset.name=custom dataset.data_dir=./data model.num_classes=3

# Experiment with different models on your data
python train.py dataset.name=custom dataset.data_dir=./data model.name=efficientnet_b0 model.num_classes=3
python train.py dataset.name=custom dataset.data_dir=./data model.name=vit_small_patch16_224 model.num_classes=3

# Use your data with any training configuration
python train.py dataset.name=custom dataset.data_dir=./my_dataset model.num_classes=20 training.epochs=100 training.batch_size=32
```

That's it! No need to write custom dataset classes or data loading code. Focus on experimentation, not boilerplate.

## 📜 License

MIT
