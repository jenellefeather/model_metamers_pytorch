"""
These variables are reused across various plots for the Feather et al. 2023 Metamers paper. 
"""
import numpy as np

## Audio model info
audio_models_to_plot = ["kell2018","cochresnet50",
                      "cochresnet50_l2_1_robust_waveform", "cochresnet50_l2_p5_robust_waveform", "cochresnet50_linf_p002_robust_waveform",
                      "cochresnet50_l2_1_random_step_waveform", "cochresnet50_linf_p002_random_step_waveform",
                      "kell2018_l2_1_robust_waveform", "kell2018_linf_p002_robust_waveform",
                      "kell2018_l2_1_random_step_waveform", "kell2018_linf_p002_random_step_waveform",
                      "cochresnet50_l2_1_robust_cochleagram", "cochresnet50_l2_p5_robust_cochleagram",
                      "cochresnet50_l2_1_random_step_cochleagram",
                      "kell2018_l2_1_robust_cochleagram", "kell2018_l2_p5_robust_cochleagram",
                      "kell2018_l2_1_random_step_cochleagram",
                     ]

audio_model_cmap = ["#000000","#004949","#009292","#ff6db6","#ffb6db",
 "#490092","#006ddb","#b66dff","#6db6ff","#b6dbff",
 "#920000","#924900","#db6d00","#24ff24","#ffff6d"]

audio_plot_color_idx = [1,10,
                10,11,12,
                10,12,
                1,2,
                1,2,
                10,11,
                10,
                1,2,
                1,
                ]

audio_plot_style = ['.','.',
              'v','^','<',
              '1','2',
              'v','<',
              '1', '2',
              'P','X',
              '4',
              'P','X',
              '4',
             ]

assert (len(np.unique(audio_plot_color_idx)) <= len(audio_model_cmap)), 'Not enough colors to plot all of the networks!'
audio_model_colors = {m:audio_model_cmap[audio_plot_color_idx[m_idx]] for m_idx, m in enumerate(audio_models_to_plot)}
audio_model_style_dict = {m:audio_plot_style[m_idx] for m_idx, m in enumerate(audio_models_to_plot)}


## Visual model info
visual_models_to_plot = [
                      'cornet_s', 'alexnet', 'vgg_19','resnet50','resnet101',
                      'resnet50_simclr', 'resnet50_moco_v2', 'resnet50_byol',
                      'resnet50_l2_3_robust', 'resnet50_linf_4_robust', 'resnet50_linf_8_robust',
                      "resnet50_random_l2_perturb", 
                      "resnet50_random_linf8_perturb",
                      'alexnet_l2_3_robust', 'alexnet_linf_8_robust',
                      "alexnet_random_l2_3_perturb", "alexnet_random_linf8_perturb",
                      'texture_shape_resnet50_trained_on_SIN', 'texture_shape_alexnet_trained_on_SIN',
                      'CLIP_resnet50', 'CLIP_ViT-B_32', 'SWSL_resnet50', 'SWSL_resnext101_32x8d', 'vision_transformer_vit_large_patch16_224',
                      'konkle_alexnetgn_ipcl_ref01_primary_model','konkle_alexnetgn_ipcl_ref12_supervised_ipcl_aug',
                     ]

visual_plot_color_idx = [
                  6,1,8,10,4, # Standard Models
                  10,11,12, # Self-Supervised Models
                  10,11,12, # Adversarially Trained ResNet50
                  10,12, # Random Perturbations ResNet50
                  1,2, # Adversarially Trained AlexNet
                  1,2, # Random Perturbations AlexNet
                  10,1, # Texture Shape Networks
                  10,5,11,4,7, # Large Dataset Models
                  1,2, # IPCL Models
]

visual_plot_line_style = ['-','-','-','-','-',
              '-','-','-',
              ':',':',':',
              '-','-',
              ':',':',
              '-','-',
              '-.','-.',
              '-', '-','-', '-','-',
              '-', '-',
             ]

visual_plot_style = ['.','.',
              '.','.','.',
              'D','D','D',
              'v','^','<',
              '1','2',
              'v','<',
              '1', '2',
              's','s', # Texture models
              '*','*','*','*','*', # Large models
              'P','X',
             ]

visual_model_cmap = ["#000000","#004949","#009292","#ff6db6","#ffb6db",
 "#490092","#006ddb","#b66dff","#6db6ff","#b6dbff",
 "#920000","#924900","#db6d00","#24ff24","#ffff6d", 'grey']

visual_model_cmap_dict = {m:visual_model_cmap[visual_plot_color_idx[m_idx]] for m_idx, m in enumerate(visual_models_to_plot)}
visual_model_style_dict = {m:visual_plot_style[m_idx] for m_idx, m in enumerate(visual_models_to_plot)}
visual_model_line_style_dict = {m:visual_plot_line_style[m_idx] for m_idx, m in enumerate(visual_models_to_plot)}
