#!/bin/bash
#SBATCH --job-name=null_spectemp
#SBATCH --output=output/null_%A_%a.out
#SBATCH --error=output/null_%A_%a.err
#SBATCH --mem=16000
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=1-999:4
#SBATCH --exclude=node093,node040,node037,node097,node098
#SBATCH --constraint=high-capacity&12GB
#SBATCH --partition=normal

module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

cp ../../../analysis_scripts/make_null_distributions.py .

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch
# python -m pdb make_null_distributions.py -N 1000
# Run 200 different starting index of 5000 to grab from different locations in the wsn dataset. 
# Training data has more than 2000000 samples so we will never be repeating pairs. 
# Spect-Temp model runs out of GPU memory if we try to allocate two models to the same GPU. 
python make_null_distributions.py -N 1000 -I $SLURM_ARRAY_TASK_ID -R 0 --no-shuffle &
python make_null_distributions.py -N 1000 -I $(($SLURM_ARRAY_TASK_ID+2)) -R 0 --no-shuffle &

wait

