# from distutils.core import setup, find_packages
from setuptools import find_packages, setup

setup(
    name='gepi-pred',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'pymongo',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'omegaconf',
        'pyyaml',
        'biopython',
        'matplotlib',
        'seaborn',
        'tqdm',
        'networkx',
        'torcheval',
        'gdown',
        'hydra-core',
        'pydantic',
        'pytest',
        'torch-scatter',
        'wandb',
        'loguru',
        'torch-geometric',
        'typing',
        'esm',
        'transformers',
        'torch-sparse',
        'tensorboard',
        'torchmetrics',
    ],
)


# pip install numpy pandas scipy scikit-learn omegaconf biopython matplotlib seaborn tqdm networkx gdown hydra-core wandb loguru typing transformers tensorboard sentencepiece
# pip install torch torchvision torchaudio pyyaml torcheval pydantic pytest torch-geometric transformers torchmetrics
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu113.html

