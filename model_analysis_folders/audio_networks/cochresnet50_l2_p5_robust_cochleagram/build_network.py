import sys
from robustness.datasets import jsinV3
from robustness.model_utils import make_and_restore_model

from model_analysis_folders.all_model_info import JSIN_PATH, MODEL_BASE_PATH
import os

# Make a custom build script for audio_rep_training_cochleagram_1/l2_p1_robust_training
def build_net(include_rep_in_model=True, 
              use_normalization_for_audio_rep=True, 
              ds_kwargs={}, 
              include_identity_sequential=False, 
              return_metamer_layers=False, 
              strict=True):

    # Build the dataset so that the number of classes and normalization 
    # is set appropriately. Not needed for metamer generation, but ds is 
    # used for eval scripts.  
    ds = jsinV3(JSIN_PATH, include_rep_in_model=include_rep_in_model, 
                audio_representation='cochleagram_1',
                use_normalization_for_audio_rep=use_normalization_for_audio_rep, 
                include_identity_sequential=include_identity_sequential, 
                **ds_kwargs) # Sequential will change the state dict names

    # Path to the network checkpoint to load
    resume_path = os.path.join(MODEL_BASE_PATH, 'audio_networks', 'pytorch_checkpoints', 'cochresnet_wsn_word_l2_p5_robust_cochleagram.pt')

    # Resnet50 Layers Used for Metamer Generation
    metamer_layers = [
         'input_after_preproc',
#          'conv1_relu1',
         'conv1_relu1_fake_relu',
#          'layer1',
         'layer1_fake_relu',
#          'layer2',
         'layer2_fake_relu',
#          'layer3',
         'layer3_fake_relu',
#          'layer4',
         'layer4_fake_relu',
         'avgpool',
         'final'
    ]

    # Restore the model
    model, _ = make_and_restore_model(arch='resnet50', 
                                      dataset=ds, 
                                      parallel=False,
                                      resume_path=resume_path,
                                      strict=strict
                                     )

    # send the model to the GPU and return it. 
    model.cuda()
    model.eval()

    if return_metamer_layers:
        return model, ds, metamer_layers
    else:
        return model, ds

def main(include_rep_in_model=True,
         use_normalization_for_audio_rep=True,
         return_metamer_layers=False,
         include_identity_sequential=False,
         strict=False,
         ds_kwargs={}):
    # This parameter is not used for this model
#     del include_identity_sequential

    if return_metamer_layers:
        model, ds, metamer_layers = build_net(include_rep_in_model=include_rep_in_model,
                                              use_normalization_for_audio_rep=use_normalization_for_audio_rep,
                                              return_metamer_layers=return_metamer_layers,
                                              strict=strict,
                                              include_identity_sequential=include_identity_sequential,
                                              ds_kwargs=ds_kwargs)
        return model, ds, metamer_layers

    else:
        model, ds = build_net(include_rep_in_model=include_rep_in_model,
                              use_normalization_for_audio_rep=use_normalization_for_audio_rep,
                              return_metamer_layers=return_metamer_layers,
                              strict=strict,
                              include_identity_sequential=include_identity_sequential,
                              ds_kwargs=ds_kwargs)
        return model, ds

if __name__== "__main__":
    main()
