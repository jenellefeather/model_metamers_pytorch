#!/usr/bin/env python

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
        "torch",
        "torchvision",
        "torchaudio",
        "h5py",
        "seaborn",
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "jupyter",
        "dill",
        "cox",
        "tables",
        "tqdm",
        "resampy",
        "tensorboardX",
        "chcochleagram @ git+https://github.com/jenellefeather/chcochleagram.git"
]

setup(
    name='model_metamers_pytorch',
    version='0.1.0',
    description='Model metamer generation in pytorch.',
    author='Jenelle Feather',
    author_email='jfeather@mit.edu',
    long_description=readme,
    install_requires=requirements,
    license="MIT license",
    keywords='adversarial, stochastic, metamers, audio, vision, robustness',
)
