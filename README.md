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
# Complete training + evaluation pipeline (recommended)
python train_and_evaluate.py                    # Train on train/val + evaluate on test
python train_and_evaluate.py +experiment=mnist_mobilenet
python train_and_evaluate.py +experiment=cifar10_resnet

# Or run training and evaluation separately:

# 1. Train only (uses train + val sets)
python train.py
python train.py +experiment=mnist_resnet

# 2. Evaluate only (uses test set)
python evaluate.py checkpoint_path=outputs/<date>/<time>/checkpoints/best-epoch=XX.ckpt
```

## 🧪 Experimentation Made Easy

The power of Padma lies in its flexibility. Mix and match any model with any dataset using simple command-line overrides:

```bash
# Experiment 1: Try EfficientNet on CIFAR-100
python train_and_evaluate.py model.model_name=efficientnet_b0 dataset=cifar100 model.num_classes=100

# Experiment 2: Compare different Vision Transformer sizes on MNIST
python train_and_evaluate.py model.model_name=vit_tiny_patch16_224 dataset=mnist
python train_and_evaluate.py model.model_name=vit_small_patch16_224 dataset=mnist
python train_and_evaluate.py model.model_name=vit_base_patch16_224 dataset=mnist

# Experiment 3: Test lightweight models for edge deployment
python train_and_evaluate.py model.model_name=mobilenetv3_small_050 training.batch_size=256
python train_and_evaluate.py model.model_name=efficientnet_b0 training.batch_size=256

# Experiment 4: Your custom dataset with any pretrained model
python train_and_evaluate.py dataset=custom dataset.data_dir=./my_data model.model_name=resnet50 model.num_classes=20

# Experiment 5: Sweep hyperparameters
python train_and_evaluate.py training.optimizer.lr=1e-4 training.batch_size=64
python train_and_evaluate.py training.optimizer.lr=1e-3 training.batch_size=128
```

No code changes needed - just configure and run. All experiments are automatically logged with full reproducibility.

**Note:** Use `train_and_evaluate.py` for complete experiments (train + test evaluation), or use `train.py` if you only need to train and validate.

## 🎯 Training & Evaluation

Padma provides three workflows for training and evaluation:

### 1. Complete Pipeline (Recommended)

**`train_and_evaluate.py`** - Trains on train/val sets, then evaluates on test set

```bash
# Complete workflow with default config
python train_and_evaluate.py

# With experiment presets
python train_and_evaluate.py +experiment=mnist_mobilenet
python train_and_evaluate.py +experiment=cifar10_resnet
python train_and_evaluate.py +experiment=imagenet_vit

# With custom parameters
python train_and_evaluate.py model.model_name=efficientnet_b0 training.epochs=30
```

This is the recommended approach as it provides a complete end-to-end pipeline: training → best checkpoint selection → test evaluation.

### 2. Training Only

**`train.py`** - Trains using train set, validates on val set, saves best checkpoint

```bash
# Basic training
python train.py

# MNIST experiments
python train.py +experiment=mnist_mobilenet    # Fastest (~1M params)
python train.py +experiment=mnist_resnet       # ResNet18

# CIFAR-10 experiments
python train.py +experiment=cifar10_resnet     # ResNet50

# Override parameters
python train.py model.model_name=resnet101
python train.py training.epochs=50 training.batch_size=64 training.optimizer.lr=1e-3
python train.py dataset=cifar100 model.num_classes=100
```

**Note:** `train.py` only trains and validates. It does **not** evaluate on the test set. Use `evaluate.py` or `train_and_evaluate.py` for test evaluation.

### 3. Evaluation Only

**`evaluate.py`** - Evaluates a trained checkpoint on the test set

```bash
# Evaluate best checkpoint
python evaluate.py checkpoint_path=outputs/<date>/<time>/checkpoints/best-epoch=XX.ckpt

# Evaluate on different dataset
python evaluate.py checkpoint_path=path/to/checkpoint.ckpt dataset=cifar10
```

**Note:** `evaluate.py` only evaluates on the **test set**. Training and validation are not performed.

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

**Note:** Test metrics are only logged when using `train_and_evaluate.py` or running `evaluate.py` separately. The `train.py` script only logs train and validation metrics.

## ⚙️ Configuration

### Directory Structure

```
configs/
├── config.yaml           # Main config
├── model/
│   └── default.yaml      # Model settings
├── dataset/
│   ├── default.yaml      # Dataset settings (defaults to MNIST)
│   ├── mnist.yaml        # MNIST-specific config (includes transforms)
│   ├── cifar10.yaml      # CIFAR-10 config (includes transforms)
│   ├── cifar100.yaml     # CIFAR-100 config (includes transforms)
│   ├── imagenet.yaml     # ImageNet config (includes transforms)
│   └── custom.yaml       # Custom dataset config (includes transforms)
├── training/
│   └── default.yaml      # Training settings
├── callbacks/
│   ├── default.yaml      # Standard callbacks
│   ├── minimal.yaml      # Minimal callback setup
│   └── early_stopping.yaml  # With early stopping enabled
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

**Callbacks** (checkpointing, early stopping, progress bar):
```bash
# Use different callback presets
python train.py callbacks=minimal          # Only checkpoint and progress bar
python train.py callbacks=early_stopping   # Enable early stopping with aggressive patience

# Enable/disable specific callbacks
python train.py callbacks.early_stopping.enabled=true callbacks.early_stopping.patience=10

# Customize callback parameters
python train.py callbacks.model_checkpoint.save_top_k=3
python train.py callbacks.early_stopping.patience=15
```

**Data Augmentation & Transforms**:

Transforms are defined directly in dataset configuration files using Hydra's `_target_` instantiation pattern. Each dataset config includes both `train_transforms` (with augmentation) and `val_transforms` (without augmentation).

```bash
# View transform configuration for a dataset
cat configs/dataset/cifar10.yaml

# Override specific transform parameters
python train.py dataset=cifar10 \
    'dataset.train_transforms[1].p=0.7'  # Change flip probability

python train.py dataset=cifar10 \
    'dataset.train_transforms[2].brightness=0.5'  # Adjust ColorJitter brightness

# Disable a specific transform by removing it from the list
python train.py dataset=cifar10 \
    'dataset.train_transforms[1]._target_=torchvision.transforms.Identity'
```

All transforms use Hydra's instantiation system with `_target_` pointing to torchvision transform classes. You can modify any transform parameter through command-line overrides or by editing the dataset YAML files directly.

## 📂 Project Structure

```
padma/
├── models/
│   ├── base.py          # Utilities (freeze, load, info)
│   └── model_factory.py # Unified ModelFactory for timm and time series models
├── datasets/
│   ├── base.py          # Dataloaders and dataset utilities
│   ├── mnist.py         # MNIST dataset
│   ├── cifar.py         # CIFAR-10/100
│   ├── imagenet.py      # ImageNet
│   └── custom.py        # Custom ImageFolder
├── trainers/
│   ├── lightning_module.py  # Lightning training module
│   └── datamodule.py        # Lightning data module
└── utils/
    ├── device.py        # Device detection
    ├── metrics.py       # Metrics tracking
    ├── callbacks.py     # Callback factory for Lightning
    ├── transforms.py    # Transform creation from Hydra configs
    └── reproducibility.py  # Seed setting
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
# Basic usage - Complete pipeline (train + test evaluation)
python train_and_evaluate.py dataset=custom dataset.data_dir=./data model.num_classes=3

# Experiment with different models on your data
python train_and_evaluate.py dataset=custom dataset.data_dir=./data model.model_name=efficientnet_b0 model.num_classes=3
python train_and_evaluate.py dataset=custom dataset.data_dir=./data model.model_name=vit_small_patch16_224 model.num_classes=3

# Use your data with any training configuration
python train_and_evaluate.py dataset=custom dataset.data_dir=./my_dataset model.num_classes=20 training.epochs=100 training.batch_size=32

# Or train only (without test evaluation)
python train.py dataset=custom dataset.data_dir=./data model.num_classes=3
```

That's it! No need to write custom dataset classes or data loading code. Focus on experimentation, not boilerplate.

## 📡 Time Series Classification

Padma now supports time series classification alongside image classification, providing the same configuration-driven workflow for temporal data.

### Available Time Series Datasets

| Dataset | Name | Channels | Timesteps | Classes | Description |
|---------|------|----------|-----------|---------|-------------|
| ECG5000 | `ecg5000` | 1 | 140 | 5 | Univariate heartbeat classification (ECG signals) |
| UCI HAR | `uci_har` | 9 | 128 | 6 | Multivariate human activity recognition (smartphone sensors) |

### Available Time Series Models

| Model | Description | Parameters | Best For |
|-------|-------------|------------|----------|
| **1D CNN** | Convolutional neural network for sequences | ~1-5M | Fast training, good baseline |
| **LSTM** | Bidirectional LSTM for sequence modeling | ~1-10M | Long-term dependencies |
| **GRU** | Bidirectional GRU (lighter than LSTM) | ~1-8M | Faster training than LSTM |
| **1D Transformer** | Attention-based sequence model | ~5-20M | State-of-the-art performance |

### Quick Start with Time Series

```bash
# ECG5000 experiments (univariate time series) - Complete pipeline
python train_and_evaluate.py +experiment=ecg5000_cnn1d          # 1D CNN (fast)
python train_and_evaluate.py +experiment=ecg5000_lstm           # LSTM
python train_and_evaluate.py +experiment=ecg5000_gru            # GRU
python train_and_evaluate.py +experiment=ecg5000_transformer    # Transformer

# UCI HAR experiments (multivariate time series) - Complete pipeline
python train_and_evaluate.py +experiment=uci_har_cnn1d          # 1D CNN
python train_and_evaluate.py +experiment=uci_har_lstm           # LSTM
python train_and_evaluate.py +experiment=uci_har_transformer    # Transformer

# Or train only (without test evaluation)
python train.py +experiment=ecg5000_cnn1d
python train.py +experiment=uci_har_lstm
```

### Custom Hyperparameters

```bash
# Adjust model architecture
python train_and_evaluate.py +experiment=ecg5000_cnn1d model.filters=[128,256,512]
python train_and_evaluate.py +experiment=ecg5000_lstm model.hidden_size=256 model.num_layers=3

# Adjust training parameters
python train_and_evaluate.py +experiment=uci_har_transformer training.batch_size=32 training.epochs=200
```

### Monitoring

Time series experiments log the same metrics as image classification (accuracy, precision, recall, F1) with the prefix `ts_` in TensorBoard.

```bash
# View time series training logs
tensorboard --logdir=outputs/
```

## 📜 License

MIT
