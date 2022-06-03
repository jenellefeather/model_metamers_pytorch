#!/bin/bash
#SBATCH --job-name=null_alexnet
#SBATCH --output=output/null_%A_%a.out
#SBATCH --error=output/null_%A_%a.err
#SBATCH --mem=64000
#SBATCH --cpus-per-task=20
#SBATCH --time=150:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --exclude=node093,node040,node094
#SBATCH --constraint=high-capacity
#SBATCH --partition=mcdermott

module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

cp ../../../analysis_scripts/make_null_distributions.py .

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch
# python -m pdb make_null_distributions.py -N 1000
# Run 5 different starting index to grab from different locations in the wsn dataset. 
# Training data has more than 2000000 samples so we will never be repeating pairs. 
python make_null_distributions.py -N 200000 -I 0 -R 0 --no-shuffle &
python make_null_distributions.py -N 200000 -I 1 -R 0 --no-shuffle &
python make_null_distributions.py -N 200000 -I 2 -R 0 --no-shuffle &
python make_null_distributions.py -N 200000 -I 3 -R 0 --no-shuffle &
python make_null_distributions.py -N 200000 -I 4 -R 0 --no-shuffle &

wait
