#!/bin/bash
#SBATCH --job-name=behavior_plot
#SBATCH --output=output/behavior_plot%j.out
#SBATCH --error=output/behavior_plot%j.err
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --time=01:20:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-30
#SBATCH --constraint=11GB
#SBATCH --exclude=node093,node094,node097,node098,node101,node102,node103,node104,node105,node106,node107,node108,node109,node110,node111,node112,node113,node114,node115,node116
#SBATCH --partition=mcdermott

# List of the networks that we want to compare
NETWORK_LIST=("alexnet" "cornet_s" "resnet50" "resnet101" "vgg_19" "alexnet_l2_3_robust" "alexnet_linf_4_robust" "alexnet_linf_8_robust" "resnet50_byol" "resnet50_simclr" "resnet50_moco_v2" "resnet50_l2_3_robust" "resnet50_linf_4_robust" "resnet50_linf_8_robust" "alexnet_random_l2_3_perturb" "alexnet_random_linf8_perturb" "resnet50_random_l2_perturb" "resnet50_random_linf8_perturb" "alexnet_early_checkpoint" "alexnet_reduced_aliasing_early_checkpoint" "vonealexnet_gaussian_noise_std4_fixed" "konkle_alexnetgn_ipcl_ref01_primary_model" "konkle_alexnetgn_ipcl_ref12_supervised_ipcl_aug" "texture_shape_alexnet_trained_on_SIN" "texture_shape_resnet50_trained_on_SIN" "CLIP_resnet50" "CLIP_ViT-B_32" "SWSL_resnet50" "SWSL_resnext101_32x8d" "vision_transformer_vit_large_patch16_224")

BUILD_MODEL=${NETWORK_LIST[$SLURM_ARRAY_TASK_ID]}
echo $BUILD_MODEL

# CD into the build model directory
cd $BUILD_MODEL

cp ../../../analysis_scripts/imagenet_network_network_comparisons.py .

mkdir output 

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.2

export CONDA_ENVS_PATH=~/my-envs:/om4/group/mcdermott/user/jfeather/conda_envs_files
source activate /om4/group/mcdermott/user/jfeather/conda_envs_files/test_metamers_public

for MET_MODEL in ${NETWORK_LIST[@]}; do
    echo $MET_MODEL
    python imagenet_network_network_comparisons.py $MET_MODEL
done

