"""
Scripts for plotting the BrainScore results vs. the metamer
recognition for the analyzed layer. 

Moved here to clean up the notebook.

See BrainscoreResultsVsMetamerRecognition.ipynb for usage
"""
import numpy as np
import pandas as pd
import os 
import glob
import math
import json
import scipy
import scipy.stats
import matplotlib.pylab as plt
import pickle as pckl
from PIL import Image
import seaborn as sns

import urllib.request, json
from scipy.io import loadmat

from notebooks.notebook_helpers import responses_network_by_layer_mat, combined_experiment_response_dictionaries, unpack_experiment_mat

def choose_metamer_layer(metamer_responses_by_layer, measured_layer):
    all_human_responses = metamer_responses_by_layer[measured_layer]
    average_metamer_recognition = np.mean(all_human_responses) # This value should vary between 0-1
    sem_metamer_recognition = np.std(all_human_responses) / np.sqrt(len(all_human_responses))
    return average_metamer_recognition, sem_metamer_recognition

def final_layer_metamer_curve(metamer_responses_by_layer, return_type):
    all_human_responses = metamer_responses_by_layer['final'] 
    if return_type=='average':
        average_metamer_recognition = np.mean(all_human_responses) # This value should vary between 0-1
        return average_metamer_recognition
    elif return_type=='sem':
        sem_metamer_recognition = np.std(all_human_responses) / np.sqrt(len(all_human_responses))
        return sem_metamer_recognition
    else:
        raise ValueError('Return type %s not recognized'%return_type)

def plot_brainscore_analysis_scatter(brainscore_df, 
                                     models_to_evaluate,
                                     model_cmap_dict,
                                     model_style_dict,
                                     combined_experiment_dict,
                                     brain_comparisons = ['V1', 'V2', 'V4', 'IT']):
    plt.figure(figsize=(3*(len(brain_comparisons)),3))
    cmap_plotting = sns.color_palette("colorblind")

    plot_idx = 1

    all_attack_averages = {model_name:[] for model_name in models_to_evaluate}

    for bs_type in brain_comparisons:
        title = 'Brain-Score: %s'%bs_type
        plt.subplot(1,len(brain_comparisons),plot_idx)
        brainscore_extracted = {}
        all_scores_list = []
        all_metamers_list = []

        for m_idx, model_name in enumerate(models_to_evaluate):
            brainscore_values = brainscore_df.loc[brainscore_df['name']==model_name]

            if len(brainscore_values)==0:
                print('Missing Brainscore for model %s'%model_name)
                continue
            brainscore_extracted[model_name] = float(brainscore_values[bs_type].values[0])

            brainscore_layer = brainscore_values[bs_type + ' best layer'].values[0]

            try:
                average_metamer_recognition, sem_metamer_recognition = choose_metamer_layer(
                    combined_experiment_dict[model_name], brainscore_layer)
            except:
                print("missing layer for %s, %s"%(bs_type, model_name))
                continue

            all_scores_list.append(brainscore_extracted[model_name])
            all_metamers_list.append(average_metamer_recognition)

            markers, caps, bars = plt.errorbar(brainscore_extracted[model_name],
                                               average_metamer_recognition, 
                                               c=model_cmap_dict[model_name], 
                                               yerr=sem_metamer_recognition,
                                               marker=model_style_dict[model_name],
                                               linestyle='',
                                               label=model_name)
            [bar.set_alpha(0.4) for bar in bars]
            [cap.set_alpha(0.4) for cap in caps]

        plt.xlim([0,1])
        plt.ylim([0,1])

        if plot_idx == len(brain_comparisons):
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, ncol=2)
        plt.title(title)
        plt.xlabel('Brain Score %s'%bs_type)
        plt.ylabel('Human Recognition')
        sns.despine()

        print('### %s ###'%bs_type)

        # For the evaluated  correlations, we Bonferroni corrected the p-values
        # for the correlation between model metamer recognizability and variance
        # explained by multiplying the p-value by 4 (the number of tests performed).
        bonferroni_correction_factor = 4
        rho_val, p_val_rho_raw = scipy.stats.spearmanr(np.array(all_scores_list), np.array(all_metamers_list))
        # Note: These were originally calculated by multiplying the rounded p_val_rho_raw
        # This caused one reported value to be 0.01 different than that reported on the paper plot. 
        # For ease of including everything together (and correctness) we calculate it here directly. 
        p_val_rho = np.min([p_val_rho_raw * bonferroni_correction_factor, 1.]) # Max out at 1.

        r_val, p_val_r_raw = scipy.stats.pearsonr(np.array(all_scores_list), np.array(all_metamers_list))
        p_val_r = np.min([p_val_r_raw * bonferroni_correction_factor, 1.])

        plt.text(0.015,0.25, 'rho=%0.2f, p=%0.2f, (raw p=%0.2f)'%(rho_val, p_val_rho, p_val_rho_raw))
        plt.text(0.015,0.15, 'R^2=%0.2f, p=%0.2f, (raw p=%0.2f)'%(r_val**2, p_val_r, p_val_r_raw))

        plot_idx+=1
