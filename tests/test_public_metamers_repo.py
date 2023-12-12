"""
Contains tests for the public model metamers repository
"""

import unittest
import numpy as np
import torch as ch
import faulthandler
from pathlib import Path
import os
faulthandler.enable()
import imp
import importlib
import sys
import robustness

# For testing metamer generation. 
from analysis_scripts import * 

# List of the networks that we want to compare
VISION_NETWORK_LIST=("alexnet", "cornet_s", "resnet50", "resnet101", "vgg_19", 
                     "alexnet_l2_3_robust", "alexnet_linf_8_robust", 
                     "resnet50_byol", "resnet50_simclr", "resnet50_moco_v2", 
                     "resnet50_l2_3_robust", "resnet50_linf_4_robust", "resnet50_linf_8_robust", 
                     "alexnet_random_l2_3_perturb", "alexnet_random_linf8_perturb", 
                     "resnet50_random_l2_perturb", "resnet50_random_linf8_perturb",
                     "alexnet_early_checkpoint", "alexnet_reduced_aliasing_early_checkpoint", "vonealexnet_gaussian_noise_std4_fixed",
                     "texture_shape_resnet50_trained_on_SIN", "texture_shape_alexnet_trained_on_SIN",
                     "CLIP_resnet50", "CLIP_ViT-B_32", "SWSL_resnet50","SWSL_resnext101_32x8d","vision_transformer_vit_large_patch16_224",
                     "konkle_alexnetgn_ipcl_ref01_primary_model", "konkle_alexnetgn_ipcl_ref12_supervised_ipcl_aug",
                     )

AUDIO_NETWORK_LIST=("cochresnet50", "kell2018", 
                    "cochresnet50_l2_1_robust_waveform", "cochresnet50_l2_p5_robust_waveform", "cochresnet50_linf_p002_robust_waveform", 
                    "kell2018_l2_1_robust_waveform", "kell2018_linf_p002_robust_waveform", 
                    "cochresnet50_l2_1_robust_cochleagram", "cochresnet50_l2_p5_robust_cochleagram",
                    "kell2018_l2_1_robust_cochleagram", "kell2018_l2_p5_robust_cochleagram", 
                    "cochresnet50_l2_1_random_step_cochleagram", "cochresnet50_l2_1_random_step_waveform", "cochresnet50_linf_p002_random_step_waveform", 
                    "kell2018_l2_1_random_step_cochleagram", "kell2018_l2_1_random_step_waveform", "kell2018_linf_p002_random_step_waveform")

class VisionNetworkTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.base_directory = os.path.dirname(os.path.abspath(__file__))
        self.model_directory_base = os.path.join(self.base_directory, 
                                                 'model_analysis_folders',
                                                 'visual_networks')

    def test_build_networks(self):
        for model in VISION_NETWORK_LIST:
            with self.subTest(model=model):
                build_network_spec = importlib.util.spec_from_file_location("build_network", 
                                        os.path.join(self.model_directory_base, model, 'build_network.py'))
                build_network = importlib.util.module_from_spec(build_network_spec)
                build_network_spec.loader.exec_module(build_network)

                model, ds, metamer_layers = build_network.main(return_metamer_layers=True)
                self.assertIsInstance(model, robustness.attacker.AttackerModel)

    def test_generate_metamers(self):
        for model in VISION_NETWORK_LIST[-1:]:
            with self.subTest(model=model):
                print(model)
                make_metamers_imagenet_16_category_val_400_only_save_metamer_layers.main(('0 -I 2 -N 2 -D model_analysis_folders/visual_networks/%s -O'%model).split())

    def test_make_null(self):
        for model in VISION_NETWORK_LIST[-1:]:
            with self.subTest(model=model):
                print(model)
                make_null_distributions.main(('-N 20 -I 0 -R 0 --shuffle -D model_analysis_folders/visual_networks/%s -O True'%model).split())

    def test_eval(self):
        for model in VISION_NETWORK_LIST[-1:]:
            with self.subTest(model=model):
               eval_natural_imagenet.main(('-D model_analysis_folders/visual_networks/%s'%model).split())
  

class AudioNetworkTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.base_directory = os.path.dirname(os.path.abspath(__file__))
        self.model_directory_base = os.path.join(self.base_directory,
                                                 'model_analysis_folders',
                                                 'audio_networks')

    def test_build_networks(self):
        for model in AUDIO_NETWORK_LIST:
            with self.subTest(model=model):
                build_network_spec = importlib.util.spec_from_file_location("build_network",
                                        os.path.join(self.model_directory_base, model, 'build_network.py'))
                build_network = importlib.util.module_from_spec(build_network_spec)
                build_network_spec.loader.exec_module(build_network)

                model, ds, metamer_layers = build_network.main(return_metamer_layers=True)
                self.assertIsInstance(model, robustness.attacker.AttackerModel)

    def test_generate_metamers(self):
        for model in AUDIO_NETWORK_LIST[-1:]:
            with self.subTest(model=model):
                # Some of the models have different cochleagram keys due to the adversarial training on the cochleagram, so allow non-strict loading (these 
                # weights are not trained and are set with model initialization, and possibly should have been omited from the state dict). 
                make_metamers_wsj400_behavior_only_save_metamer_layers.main(('0 -I 2 -N 2 -D model_analysis_folders/audio_networks/%s -O'%model).split())
                pass

    def test_make_null(self):
        for model in AUDIO_NETWORK_LIST[-1:]:
            with self.subTest(model=model):
                # Some of the models have different cochleagram keys due to the adversarial training on the cochleagram, so allow non-strict loading (these
                # weights are not trained and are set with model initialization, and possibly should have been omited from the state dict).
                # pass
                make_null_distributions.main(('-N 20 -I 0 -R 0 --no-shuffle -D model_analysis_folders/audio_networks/%s -O True'%model).split())

    def test_eval(self):
        for model in AUDIO_NETWORK_LIST[-1:]:
            with self.subTest(model=model):
               eval_natural_jsinv3.main(('-D model_analysis_folders/audio_networks/%s'%model).split())
                

if __name__ == "__main__":
    unittest.main()
    
