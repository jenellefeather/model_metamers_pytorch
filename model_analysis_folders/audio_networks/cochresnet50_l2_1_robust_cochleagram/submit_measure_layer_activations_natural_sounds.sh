#!/bin/bash
#SBATCH --job-name=l2_p3_regression
#SBATCH --output=./output/net_word%A_%a.out
#SBATCH --error=./output/net_word%A_%a.err
#SBATCH --gres=gpu:GEFORCEGTX1080TI:1
#SBATCH --mem=8000
#SBATCH --time=03:30:00
#SBATCH --cpus-per-task=2

module add openmind/singularity
module add openmind/cuda/9.1
module add openmind/cudnn/9.1-7.0.5

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch

cp ../../../analysis_scripts/measure_layer_activations_165_natural_sounds_pytorch.py .

python measure_layer_activations_165_natural_sounds_pytorch.py 'build_network.py'
