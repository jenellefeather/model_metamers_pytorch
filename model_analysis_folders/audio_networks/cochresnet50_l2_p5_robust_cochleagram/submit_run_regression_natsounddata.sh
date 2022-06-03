#!/bin/bash
#SBATCH --job-name=natsound_regressions_l21
#SBATCH --output=output/%x_%A_%a.out
#SBATCH --error=output/%x_%A_%a.err
#SBATCH --mem=16000
#SBATCH --array=0-12
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=mcdermott

module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

cp /om4/group/mcdermott/user/jfeather/projects/robust_audio_networks/robustness/analysis_scripts_metamers_paper/run_regressions_all_voxels_om_natsounddata.py .
export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch

python run_regressions_all_voxels_om_natsounddata.py $SLURM_ARRAY_TASK_ID $i 'natsound_activations.h5' -Z
