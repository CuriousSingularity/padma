# Padma

A modular PyTorch training framework using Hydra for configuration management and timm for pretrained model weights.

## Installation

```bash
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Quick Start

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

## Training

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

### Available Models

| Architecture | Model Names | Params |
|--------------|-------------|--------|
| MobileNet | `mobilenetv3_small_050`, `mobilenetv3_small_100`, `mobilenetv3_large_100` | 1-5M |
| ResNet | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` | 11-60M |
| EfficientNet | `efficientnet_b0` to `efficientnet_b7`, `efficientnetv2_s/m/l` | 5-66M |
| ConvNeXt | `convnext_tiny`, `convnext_small`, `convnext_base` | 28-89M |
| ViT | `vit_tiny_patch16_224`, `vit_small_patch16_224`, `vit_base_patch16_224` | 6-86M |

### Available Datasets

| Dataset | Name | Classes |
|---------|------|---------|
| MNIST | `mnist` | 10 |
| CIFAR-10 | `cifar10` | 10 |
| CIFAR-100 | `cifar100` | 100 |
| ImageNet | `imagenet` | 1000 |
| Custom | `custom` | - |

## Evaluation

After training, evaluate the model:

```bash
# Evaluate best checkpoint
python evaluate.py checkpoint_path=outputs/<date>/<time>/checkpoints/best.pt

# Evaluate last checkpoint
python evaluate.py checkpoint_path=outputs/<date>/<time>/checkpoints/last.pt

# Evaluate on different dataset
python evaluate.py checkpoint_path=path/to/best.pt dataset.name=cifar10
```

Evaluation outputs:
- Validation and test set metrics (accuracy, precision, recall, F1)
- Per-class accuracy breakdown
- Results saved to `evaluation_results.json`

## Monitoring

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

## Configuration

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

## Project Structure

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

## Custom Dataset

Organize your data as:

```
data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/  (optional)
    ├── class1/
    ├── class2/
    └── ...
```

Then train:

```bash
python train.py dataset.name=custom dataset.data_dir=./data model.num_classes=<num_classes>
```

## License

MIT
