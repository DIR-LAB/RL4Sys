# setup.py

from setuptools import setup, find_packages

setup(
    name="rl4sys",
    version="0.1.0",
    description="A Framework for Reinforcement Learning Optimization",
    author="RL4Sys Team",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "pyzmq",
        "tensorboard",
        "gym",
        "scipy",
        "pygame",
        "seaborn",
        "tensorflow",
        "joblib",
        "psutil",
        "tqdm",
        "box2d",
        "gymnasium",
        "grpcio",
        "grpcio-tools",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)