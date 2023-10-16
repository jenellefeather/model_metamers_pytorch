#!/bin/bash
#SBATCH --job-name=behavior_plot
#SBATCH --output=output/behavior_plot%A_%a.out
#SBATCH --error=output/behavior_plot%A_%a.out
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --array=0-1
#SBATCH --constraint=11GB
#SBATCH --exclude=node088,node114,node107,node112
#SBATCH --partition=normal

# List of the networks that we want to compare
NETWORK_LIST=("konkle_alexnetgn_ipcl_ref01_primary_model" "konkle_alexnetgn_ipcl_ref12_supervised_ipcl_aug" "texture_shape_alexnet_trained_on_SIN" "texture_shape_resnet50_trained_on_SIN" "CLIP_resnet50" "CLIP_ViT-B_32" "SWSL_resnet50" "SWSL_resnext101_32x8d" "vision_transformer_vit_large_patch16_224" "alexnet" "cornet_s" "resnet50" "resnet101" "vgg_19" "alexnet_l2_3_robust" "alexnet_linf_4_robust" "alexnet_linf_8_robust" "resnet50_byol" "resnet50_simclr" "resnet50_moco_v2" "resnet50_l2_3_robust" "resnet50_linf_4_robust" "resnet50_linf_8_robust" "alexnet_random_l2_3_perturb" "alexnet_random_linf8_perturb" "resnet50_random_l2_perturb" "resnet50_random_linf8_perturb" "alexnet_early_checkpoint" "alexnet_reduced_aliasing_early_checkpoint" "vonealexnet_gaussian_noise_std4_fixed")

BUILD_MODEL=${NETWORK_LIST[$SLURM_ARRAY_TASK_ID]}
echo $BUILD_MODEL

# CD into the build model directory
cd $BUILD_MODEL

cp ../../analysis_scripts/eval_adv_imagenet_with_robustness_lib.py .

module add openmind/cudnn/11.5-v8.3.3.40
module add openmind/cuda/11.3
source activate /om/user/jfeather/.conda/envs/model_metamers_pytorch_update_pytorch

for RAND_SEED in 3 1 0 2 4; do 
    echo $RAND_SEED
    python eval_adv_imagenet_with_robustness_lib.py -R $RAND_SEED -N 1024 -I 64 -B 16 -T '2' -D 4 -U 2 -E '0,3'
    python eval_adv_imagenet_with_robustness_lib.py -R $RAND_SEED -N 1024 -I 64 -B 16 -T 'inf' -D 4 -U 2 -E '0,4/255'
done
