from setuptools import setup

setup(
    name='model_metamers_pytorch',
    version='0.1.1',
    description='Model metamer generation in pytorch.',
    author='Jenelle Feather',
    author_email='jfeather@mit.edu',
    license='MIT',
    install_requires=[
        'chcochleagram @ git+https://git@github.com/jenellefeather/chcochleagram.git',
        'numpy',
        'matplotlib',
        'jupyter',
        'torch',
        'scipy',
        'tqdm', 'grpcio', 'psutil', 'gitpython','py3nvml', 'cox',
        'scikit-learn', 'seaborn', 'torchvision', 'pandas',
        'numpy', 'scipy', 'GPUtil', 'dill', 'tensorboardX', 'tables',
        'matplotlib','torchaudio','resampy',
    ],
)
