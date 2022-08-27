import sys
from robustness.datasets import jsinV3
from robustness.model_utils import make_and_restore_model

from model_analysis_folders.all_model_info import JSIN_PATH, MODEL_BASE_PATH
import os

def build_net(include_rep_in_model=True, 
              use_normalization_for_audio_rep=True, 
              ds_kwargs={}, 
              include_identity_sequential=False, 
              return_metamer_layers=False, 
              strict=True):

    # Build the dataset so that the number of classes and normalization 
    # is set appropriately. Not needed for metamer generation, but ds is 
    # used for eval scripts.  
    # Audio models include an additional `include_rep_in_model` flag, 
    # which will use the cochleagram as part of the model (generating adverarial 
    # examples at the wavefrm) or as part of the input representation (generating
    # adversarial examples at the cochleagram)
    # Metamers should be generated at the waveform (include_rep_in_model=True)
    ds = jsinV3(JSIN_PATH, include_rep_in_model=include_rep_in_model, 
                audio_representation='cochleagram_1', # specify the input audio representation. 
                use_normalization_for_audio_rep=use_normalization_for_audio_rep, 
                include_identity_sequential=include_identity_sequential, 
                **ds_kwargs) # Sequential will change the state dict names

    # TODO: If not included in the architecture file, specify the model checkpoint to use for loading.
    resume_path = None

    # List of keys in the `all_outputs` dictionary.
    metamer_layers = [
         'input_after_preproc', # Used for visualization purposes, typically the cochleagram
         ## TODO: Add additional layers here, matching the layers used in the architecture dictionary
         'final' # classifier layer if it exists. Otherwise, a placeholder.
    ]

    # TODO: Specify the model name and restore model
    # Load the model architeccture from the robustness repo. Look in robustness audio_models
    # for examples. These should have the ability to return the `all_outputs` dictionary.
    model, _ = make_and_restore_model(arch='<MODEL_ARCHITECTURE_NAME>', 
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
         use_normalization_for_audio_rep=False,
         return_metamer_layers=False,
         include_identity_sequential=False,
         strict=True,
         ds_kwargs={}):

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
