from setuptools import setup, find_packages

setup(
    name='model_metamers_pytorch',
    version='0.1.0',
    description='Model metamer generation in pytorch.',
    author='Jenelle Feather',
    author_email='jfeather@mit.edu',
    license='MIT',
    packages=find_packages(include=['robustness', 'assets', 'analysis_scripts', 'model_analysis_folders']),
    install_requires=[
        'chcochleagram @ git+ssh://git@github.com/jenellefeather/chcochleagram.git'
        'numpy',
        'matplotlib',
        'jupyter',
        'torch',
        'scipy',
        'tqdm', 'grpcio', 'psutil', 'gitpython','py3nvml', 'cox',
        'scikit-learn', 'seaborn', 'torchvision', 'pandas',
        'numpy', 'scipy', 'GPUtil', 'dill', 'tensorboardX', 'tables',
        'matplotlib','torchaudio'
    ],
)
