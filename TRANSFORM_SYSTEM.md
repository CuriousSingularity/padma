# Hydra-Based PyTorch Transform System

## Overview

I've implemented a flexible, YAML-driven transform system for Padma using Hydra's `hydra.utils.instantiate` functionality. This allows you to define all PyTorch augmentation transforms in YAML configuration files and compose them dynamically at runtime.

## What Was Implemented

### 1. Core Utilities (`padma/utils/transforms.py`)

A new utility module providing:

- **`create_transforms_from_config()`** - Creates torchvision.transforms.Compose from Hydra config
- **`create_transform_from_dict()`** - Creates single transform from name and args
- **`create_transforms_from_list()`** - Creates transforms from Python dictionaries
- **`get_transform_info()`** - Returns human-readable transform description

### 2. Transform Configuration Files (`configs/transformation/`)

Six pre-configured transform presets:

#### General Purpose
- **`train.yaml`** - Standard ImageNet-style training transforms (224×224)
- **`val.yaml`** - Standard validation/test transforms

#### Augmentation Intensity
- **`train_light.yaml`** - Minimal augmentations (crop + flip)
- **`train_heavy.yaml`** - Aggressive augmentations (rotation, color jitter, random erasing)

#### Dataset-Specific
- **`mnist_train.yaml`** - MNIST training (28×28, grayscale)
- **`mnist_val.yaml`** - MNIST validation

#### Experiment Configs
- **`cifar10_experiment.yaml`** - Complete CIFAR-10 pipeline (with dataset-specific normalization)
- **`imagenet_experiment.yaml`** - ImageNet pipeline with RandAugment

### 3. Example Code

- **`examples/transforms_demo.py`** - Interactive demo showing:
  - Loading transforms from YAML
  - Applying to images
  - Using different presets
  - Manual transform creation

- **`examples/dataset_with_hydra_transforms.py`** - Integration examples:
  - `create_cifar10_with_hydra_transforms()` - CIFAR-10 with Hydra transforms
  - `create_mnist_with_hydra_transforms()` - MNIST with Hydra transforms
  - Shows how to integrate with existing dataset factory pattern

### 4. Documentation

- **`configs/transformation/README.md`** - Comprehensive guide covering:
  - All available presets
  - Usage examples
  - Parameter reference
  - Common patterns
  - Troubleshooting

### 5. Testing

- **`test_transforms.py`** - Verification tests:
  - Basic transform creation
  - Augmentation transforms
  - Transforms from Python lists
  - Grayscale transforms
  - YAML config loading

## How It Works

### YAML Configuration Format

Each transform uses the `_target_` pattern to specify which class to instantiate:

```yaml
transforms:
  - _target_: torchvision.transforms.RandomResizedCrop
    size: 224
    scale: [0.08, 1.0]
    ratio: [0.75, 1.333]
    interpolation: 3

  - _target_: torchvision.transforms.RandomHorizontalFlip
    p: 0.5

  - _target_: torchvision.transforms.ColorJitter
    brightness: 0.4
    contrast: 0.4
    saturation: 0.4
    hue: 0.1

  - _target_: torchvision.transforms.ToTensor

  - _target_: torchvision.transforms.Normalize
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

### Python Usage

```python
from omegaconf import DictConfig
from padma.utils.transforms import create_transforms_from_config

# In your dataset factory
def create_dataset(cfg: DictConfig):
    # Create transforms from config
    train_transform = create_transforms_from_config(
        cfg.transformation.train_transforms
    )

    # Use with PyTorch dataset
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        transform=train_transform
    )

    return dataset
```

### Command Line Usage

```bash
# Use a preset
python train.py +transformation=train
python train.py +transformation=train_heavy
python train.py +transformation=mnist_train

# Use experiment config (includes both train and val)
python train.py +transformation=cifar10_experiment

# Override parameters
python train.py +transformation=train \\
    transformation.transforms[1].p=0.7 \\
    transformation.transforms[2].brightness=0.5
```

## Key Benefits

1. **Zero Code Changes for Experimentation**
   - All transform parameters in YAML
   - Modify via CLI without touching code

2. **Type-Safe Configuration**
   - Hydra validates all parameters
   - IDE support for YAML editing

3. **Fully Composable**
   - Mix and match transform presets
   - Override specific parameters
   - Create custom configs easily

4. **Reproducible**
   - Config automatically saved with outputs
   - Exact transform pipeline preserved

5. **All torchvision Transforms Supported**
   - Works with any torchvision.transforms class
   - Supports custom transforms too

## Usage Examples

### Example 1: Basic Training

```bash
# Use standard training transforms
python train.py dataset=cifar10 +transformation=train
```

### Example 2: Heavy Augmentation

```bash
# When model is overfitting, use aggressive augmentation
python train.py dataset=cifar10 +transformation=train_heavy
```

### Example 3: Custom Parameters

```bash
# Fine-tune ColorJitter values
python train.py +transformation=train \\
    transformation.transforms[2].brightness=0.3 \\
    transformation.transforms[2].contrast=0.3
```

### Example 4: MNIST-Specific

```bash
# Use MNIST-specific transforms (grayscale, 28×28)
python train.py dataset=mnist +transformation=mnist_train
```

### Example 5: Complete Experiment

```bash
# Use pre-configured experiment (train + val transforms)
python train.py +transformation=cifar10_experiment \\
    model=resnet18 \\
    training.epochs=100
```

## File Structure

```
configs/transformation/
├── README.md                      # Comprehensive documentation
├── train.yaml                     # Standard training transforms
├── val.yaml                       # Validation/test transforms
├── train_light.yaml              # Minimal augmentations
├── train_heavy.yaml              # Aggressive augmentations
├── mnist_train.yaml              # MNIST training
├── mnist_val.yaml                # MNIST validation
├── cifar10_experiment.yaml       # CIFAR-10 complete pipeline
└── imagenet_experiment.yaml      # ImageNet complete pipeline

padma/utils/
└── transforms.py                  # Core utility functions

examples/
├── transforms_demo.py            # Interactive demo
└── dataset_with_hydra_transforms.py  # Dataset integration examples

test_transforms.py                 # Verification tests
```

## Creating Custom Transforms

### Step 1: Create YAML Config

Create `configs/transformation/my_custom.yaml`:

```yaml
transforms:
  - _target_: torchvision.transforms.Resize
    size: 256

  - _target_: torchvision.transforms.RandomCrop
    size: 224
    padding: 10

  - _target_: torchvision.transforms.ToTensor

  - _target_: torchvision.transforms.Normalize
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

### Step 2: Use It

```bash
python train.py +transformation=my_custom
```

That's it! No code changes needed.

## Advanced Usage

### Separate Train/Val Configs

For experiment configs with different train and validation transforms:

```yaml
# configs/transformation/my_experiment.yaml
train_transforms:
  - _target_: torchvision.transforms.RandomHorizontalFlip
  - _target_: torchvision.transforms.ToTensor

val_transforms:
  - _target_: torchvision.transforms.ToTensor
```

### Dataset-Specific Normalization

```yaml
# CIFAR-10 normalization
- _target_: torchvision.transforms.Normalize
  mean: [0.4914, 0.4822, 0.4465]  # CIFAR-10 specific
  std: [0.2470, 0.2435, 0.2616]

# ImageNet normalization
- _target_: torchvision.transforms.Normalize
  mean: [0.485, 0.456, 0.406]  # ImageNet
  std: [0.229, 0.224, 0.225]
```

### Auto-Augmentation

```yaml
# RandAugment
- _target_: torchvision.transforms.RandAugment
  num_ops: 2
  magnitude: 9
  interpolation: 3

# AugMix
- _target_: torchvision.transforms.AugMix
  severity: 3
  mixture_width: 3

# AutoAugment
- _target_: torchvision.transforms.AutoAugment
  policy: 1  # 0=ImageNet, 1=CIFAR10, 2=SVHN
```

## Testing

To verify the implementation works:

```bash
# Run tests (requires torch, torchvision, omegaconf, hydra)
python test_transforms.py

# Try the demo
python examples/transforms_demo.py +transformation=train

# Test with actual training (if environment is set up)
python train.py +transformation=cifar10_experiment
```

## Migration from Existing System

The old `ImageAugmentation` class-based system in `padma/datasets/base.py` is still intact and working. To migrate to the new system:

1. **Keep existing code working** - No breaking changes
2. **Test new system** - Try it with your datasets
3. **Gradually migrate** - Convert one dataset at a time
4. **Remove old system** - Once all datasets migrated

Example migration:

**Old way:**
```python
from padma.datasets.base import get_transforms
train_transform = get_transforms(cfg, is_training=True)
```

**New way:**
```python
from padma.utils.transforms import create_transforms_from_config
train_transform = create_transforms_from_config(cfg.transformation.train_transforms)
```

## Next Steps

1. **Install dependencies** (if not already):
   ```bash
   pip install torch torchvision omegaconf hydra-core
   ```

2. **Try the demo**:
   ```bash
   python examples/transforms_demo.py +transformation=train
   ```

3. **Read the docs**:
   - `configs/transformation/README.md` - Detailed usage guide
   - `examples/dataset_with_hydra_transforms.py` - Integration examples

4. **Experiment**:
   - Create custom transform configs
   - Try different augmentation intensities
   - Override parameters via CLI

5. **Integrate with your datasets**:
   - Use examples as templates
   - Update dataset factories to use new system
   - Test thoroughly before deploying

## Support

- **Documentation**: `configs/transformation/README.md`
- **Examples**: `examples/transforms_demo.py` and `examples/dataset_with_hydra_transforms.py`
- **Tests**: `test_transforms.py`

## References

- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [Hydra Instantiate](https://hydra.cc/docs/advanced/instantiate_objects/overview/)
- [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)
- [RandAugment Paper](https://arxiv.org/abs/1909.13719)
