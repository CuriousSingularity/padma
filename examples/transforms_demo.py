"""Demo script showing how to use Hydra-based transform configuration.

This script demonstrates:
1. Loading transform configs from YAML files
2. Creating transforms using hydra.utils.instantiate
3. Applying transforms to images
4. Using different transform presets (train, val, light, heavy)

Usage:
    python examples/transforms_demo.py
    python examples/transforms_demo.py transformation=train_heavy
    python examples/transforms_demo.py transformation=mnist_train
"""

import hydra
from omegaconf import DictConfig
import torch
from PIL import Image
import numpy as np

from padma.utils.transforms import create_transforms_from_config, get_transform_info


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function demonstrating transform usage."""

    print("=" * 80)
    print("PyTorch Transforms Demo - Hydra Instantiation")
    print("=" * 80)

    # Example 1: Load transforms from configuration
    print("\n[Example 1] Loading transforms from YAML config\n")

    # Check if transformation config exists in the config
    if "transformation" not in cfg:
        print("⚠️  No transformation config found.")
        print("To use this demo, add transformation configs to your main config.")
        print("\nYou can manually load a transformation config:")
        print("  python examples/transforms_demo.py +transformation=train")
        print("  python examples/transforms_demo.py +transformation=train_heavy")
        return

    # Create transforms from config
    if "transforms" in cfg.transformation:
        print(f"📋 Loading transforms from config...")
        transform = create_transforms_from_config(cfg.transformation.transforms)
        print(f"\n✅ Created transform pipeline:")
        print(get_transform_info(transform))
    else:
        print("⚠️  'transforms' key not found in transformation config")
        return

    # Example 2: Apply transform to a sample image
    print("\n" + "=" * 80)
    print("[Example 2] Applying transforms to a sample image")
    print("=" * 80)

    # Create a random RGB image (224x224)
    print("\n📷 Creating sample RGB image (224x224)...")
    sample_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    print(f"Original image: {sample_image.size}, mode: {sample_image.mode}")

    # Apply transform
    print("\n🔄 Applying transforms...")
    try:
        transformed = transform(sample_image)
        print(f"✅ Transformed tensor shape: {transformed.shape}")
        print(f"   Tensor dtype: {transformed.dtype}")
        print(f"   Value range: [{transformed.min():.3f}, {transformed.max():.3f}]")
    except Exception as e:
        print(f"❌ Error applying transform: {e}")

    # Example 3: Show how to use different transform configs
    print("\n" + "=" * 80)
    print("[Example 3] Available transform configurations")
    print("=" * 80)

    print("\nYou can use different transform presets:")
    print("  • train.yaml          - Standard training transforms")
    print("  • val.yaml            - Validation/test transforms")
    print("  • train_light.yaml    - Light augmentations")
    print("  • train_heavy.yaml    - Heavy augmentations")
    print("  • mnist_train.yaml    - MNIST-specific training")
    print("  • mnist_val.yaml      - MNIST-specific validation")

    print("\nTo use a specific preset:")
    print("  python examples/transforms_demo.py +transformation=train")
    print("  python examples/transforms_demo.py +transformation=train_heavy")
    print("  python examples/transforms_demo.py +transformation=mnist_train")

    # Example 4: Manual transform creation (without Hydra)
    print("\n" + "=" * 80)
    print("[Example 4] Creating transforms manually (without config files)")
    print("=" * 80)

    from padma.utils.transforms import create_transforms_from_list

    manual_transforms = [
        {"name": "Resize", "args": {"size": 256}},
        {"name": "CenterCrop", "args": {"size": 224}},
        {"name": "ToTensor"},
        {"name": "Normalize", "args": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
    ]

    print("\n📋 Creating transforms programmatically...")
    manual_transform = create_transforms_from_list(manual_transforms)
    print(f"\n✅ Created transform pipeline:")
    print(get_transform_info(manual_transform))

    # Apply manual transform
    print("\n🔄 Applying manual transforms...")
    try:
        transformed_manual = manual_transform(sample_image)
        print(f"✅ Transformed tensor shape: {transformed_manual.shape}")
    except Exception as e:
        print(f"❌ Error applying transform: {e}")

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
