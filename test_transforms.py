"""Quick test script to verify Hydra transform instantiation works correctly.

This script tests the transform utilities without requiring a full training setup.
Run this to verify the implementation is working.

Usage:
    python test_transforms.py
"""

import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from padma.utils.transforms import (
    create_transforms_from_config,
    create_transforms_from_list,
    get_transform_info,
)


def test_basic_transform_creation():
    """Test creating transforms from OmegaConf."""
    print("=" * 80)
    print("Test 1: Basic Transform Creation from Config")
    print("=" * 80)

    # Create a simple transform config
    config = OmegaConf.create(
        {
            "transforms": [
                {
                    "_target_": "torchvision.transforms.Resize",
                    "size": 256,
                },
                {
                    "_target_": "torchvision.transforms.CenterCrop",
                    "size": 224,
                },
                {
                    "_target_": "torchvision.transforms.ToTensor",
                },
                {
                    "_target_": "torchvision.transforms.Normalize",
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
            ]
        }
    )

    # Create transform
    transform = create_transforms_from_config(config.transforms)
    print(f"\n✅ Created transform pipeline:")
    print(get_transform_info(transform))

    # Test on image
    test_image = Image.fromarray(
        np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    )
    print(f"\n📷 Test image: {test_image.size}")

    result = transform(test_image)
    print(f"✅ Output tensor: {result.shape}, dtype: {result.dtype}")

    assert result.shape == (3, 224, 224), "Output shape mismatch"
    print("✅ Test 1 PASSED\n")


def test_augmentation_transforms():
    """Test augmentation transforms."""
    print("=" * 80)
    print("Test 2: Augmentation Transforms")
    print("=" * 80)

    config = OmegaConf.create(
        {
            "transforms": [
                {
                    "_target_": "torchvision.transforms.RandomResizedCrop",
                    "size": 224,
                    "scale": [0.08, 1.0],
                },
                {
                    "_target_": "torchvision.transforms.RandomHorizontalFlip",
                    "p": 0.5,
                },
                {
                    "_target_": "torchvision.transforms.ColorJitter",
                    "brightness": 0.4,
                    "contrast": 0.4,
                    "saturation": 0.4,
                    "hue": 0.1,
                },
                {
                    "_target_": "torchvision.transforms.ToTensor",
                },
            ]
        }
    )

    transform = create_transforms_from_config(config.transforms)
    print(f"\n✅ Created augmentation pipeline:")
    print(get_transform_info(transform))

    # Test on image
    test_image = Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    )

    result = transform(test_image)
    print(f"\n✅ Output tensor: {result.shape}, dtype: {result.dtype}")

    assert result.shape == (3, 224, 224), "Output shape mismatch"
    assert result.dtype == torch.float32, "Output dtype mismatch"
    print("✅ Test 2 PASSED\n")


def test_transforms_from_list():
    """Test creating transforms from Python list."""
    print("=" * 80)
    print("Test 3: Creating Transforms from Python List")
    print("=" * 80)

    transform_defs = [
        {"name": "Resize", "args": {"size": 256}},
        {"name": "CenterCrop", "args": {"size": 224}},
        {"name": "ToTensor"},
        {"name": "Normalize", "args": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
    ]

    transform = create_transforms_from_list(transform_defs)
    print(f"\n✅ Created transform pipeline:")
    print(get_transform_info(transform))

    # Test on image
    test_image = Image.fromarray(
        np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    )

    result = transform(test_image)
    print(f"\n✅ Output tensor: {result.shape}, dtype: {result.dtype}")

    assert result.shape == (3, 224, 224), "Output shape mismatch"
    print("✅ Test 3 PASSED\n")


def test_grayscale_transforms():
    """Test transforms for grayscale images (MNIST-style)."""
    print("=" * 80)
    print("Test 4: Grayscale Transforms (MNIST)")
    print("=" * 80)

    config = OmegaConf.create(
        {
            "transforms": [
                {
                    "_target_": "torchvision.transforms.ToTensor",
                },
                {
                    "_target_": "torchvision.transforms.Normalize",
                    "mean": [0.1307],
                    "std": [0.3081],
                },
            ]
        }
    )

    transform = create_transforms_from_config(config.transforms)
    print(f"\n✅ Created grayscale pipeline:")
    print(get_transform_info(transform))

    # Test on grayscale image
    test_image = Image.fromarray(
        np.random.randint(0, 255, (28, 28), dtype=np.uint8), mode="L"
    )
    print(f"\n📷 Test image: {test_image.size}, mode: {test_image.mode}")

    result = transform(test_image)
    print(f"✅ Output tensor: {result.shape}, dtype: {result.dtype}")

    assert result.shape == (1, 28, 28), "Output shape mismatch"
    print("✅ Test 4 PASSED\n")


def test_loading_yaml_config():
    """Test loading actual YAML config files."""
    print("=" * 80)
    print("Test 5: Loading YAML Config Files")
    print("=" * 80)

    # Try to load a config file
    try:
        config = OmegaConf.load("configs/transformation/train.yaml")
        print(f"\n✅ Loaded config from YAML:")
        print(OmegaConf.to_yaml(config))

        transform = create_transforms_from_config(config.transforms)
        print(f"\n✅ Created transform pipeline from YAML:")
        print(get_transform_info(transform))

        # Test it
        test_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        result = transform(test_image)
        print(f"\n✅ Output tensor: {result.shape}, dtype: {result.dtype}")
        print("✅ Test 5 PASSED\n")

    except FileNotFoundError:
        print("⚠️  Config file not found - skipping test")
        print("   (This is OK if running from a different directory)")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING HYDRA TRANSFORM UTILITIES")
    print("=" * 80 + "\n")

    try:
        test_basic_transform_creation()
        test_augmentation_transforms()
        test_transforms_from_list()
        test_grayscale_transforms()
        test_loading_yaml_config()

        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe Hydra transform system is working correctly.")
        print("You can now use it in your training pipeline.")
        print("\nNext steps:")
        print("  1. Try the demo: python examples/transforms_demo.py +transformation=train")
        print("  2. Check the docs: configs/transformation/README.md")
        print("  3. Integrate with datasets: examples/dataset_with_hydra_transforms.py")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
