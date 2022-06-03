#!/bin/bash
#SBATCH --job-name=behavior_plot
#SBATCH --output=output/behavior_plot%j.out
#SBATCH --error=output/behavior_plot%j.err
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-20
#SBATCH --constraint=11GB
#SBATCH --partition=normal

# List of the networks that we want to compare
NETWORK_LIST=("cochresnet50" "kell2018" "cochresnet50_l2_1_robust_waveform" "cochresnet50_l2_p5_robust_waveform" "cochresnet50_linf_p002_robust_waveform" "kell2018_l2_1_robust_waveform" "kell2018_linf_p002_robust_waveform" "cochresnet50_l2_1_robust_cochleagram" "cochresnet50_l2_p5_robust_cochleagram" "kell2018_l2_1_robust_cochleagram" "kell2018_l2_p5_robust_cochleagram" "cochresnet50_l2_1_random_step_cochleagram" "cochresnet50_l2_1_random_step_waveform" "cochresnet50_linf_p002_random_step_waveform" "kell2018_l2_1_random_step_cochleagram" "kell2018_l2_1_random_step_waveform" "kell2018_linf_p002_random_step_waveform")

BUILD_MODEL=${NETWORK_LIST[$SLURM_ARRAY_TASK_ID]}
echo $BUILD_MODEL

# CD into the build model directory
cd $BUILD_MODEL

module add openmind/miniconda/2020-01-29-py3.7
module add openmind/cudnn/9.1-7.0.5
module add openmind/cuda/9.1

cp ../../../analysis_scripts/jsinv3_word_network_network_comparisons.py .

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/pytorch

for MET_MODEL in ${NETWORK_LIST[@]}; do
    echo $MET_MODEL
    python jsinv3_word_network_network_comparisons.py $MET_MODEL
done

