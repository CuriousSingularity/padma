from setuptools import setup, find_packages

setup(
    name="padma",
    version="0.1.0",
    description="A modular PyTorch training framework with Hydra configuration",
    python_requires=">=3.12",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "lightning>=2.0.0",
        "timm>=0.9.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "tensorboard>=2.14.0",
        "torchmetrics>=1.0.0",
        "pillow>=9.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "rich>=13.0.0",
        "onnx>=1.15.0",
        "onnxscript>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "onnx-runtime": [
            "onnxruntime>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "padma-train=train:main",
            "padma-evaluate=evaluate:main",
        ],
    },
)
