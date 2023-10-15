"""
Scripts for running the best layer analysis for the fMRI data and plotting it.
Moved here to clean up the notebook. 

See fMRIBestLayerVsMetamerRecognition.ipynb for usage
"""
import numpy as np
import pickle

import os

import matplotlib.pylab as plt

from model_analysis_folders import all_model_info, color_model_info

import scipy.stats

import seaborn as sns


def split_voxel_data_by_participant(voxel_data, voxel_meta):
    subj_idx = voxel_meta['subj_idx']
    all_participants = []
    participant_ids = np.unique(subj_idx)
    for participant_id in participant_ids:
        all_participants.append(voxel_data[:,subj_idx==participant_id,:])
    return participant_ids, all_participants

def split_roi_data_by_participant(voxel_data, voxel_meta, roi_masks):
    subj_idx = voxel_meta['subj_idx']
    all_participants = []
    participant_ids = np.unique(subj_idx)
    for participant_id in participant_ids:
        all_participants.append({roi_name: voxel_data[:,(np.logical_and(subj_idx==participant_id,  
                                                         roi_masks['masks'][:,roi_idx].astype(bool))),:] 
                                 for roi_idx, roi_name in enumerate(roi_masks['mask_names'])})
        all_participants[-1]['all_aud_voxels'] = voxel_data[:,subj_idx==participant_id,:]
    return participant_ids, all_participants


def load_all_voxel_predictions(all_networks_and_layers, model_name,
                               spectemp_filter_layer='avgpool'):
    """
    Loads in the voxel predictions for a given model. 
    
    Inputs:
        all_networks_and_layers (dict): contains all of the info about the models
        model_name (string): contains the model name, used to load and save
    """
    all_conv_layers = all_networks_and_layers[model_name]['layers']
    if (model_name == 'spectemp_filters') and (spectemp_filter_layer is not None):
        all_conv_layers = [spectemp_filter_layer]
    neural_location = os.path.join(all_networks_and_layers[model_name]['location'], 
                                   all_networks_and_layers[model_name]['neural_regression_folder'])
    neural_analysis_string = all_networks_and_layers[model_name]['neural_analysis_string']
    all_r2 = {}
    all_alphas = {}
    n_layers = len(all_conv_layers)
    
    all_voxel_info = {}
    
    init_conv = True
        
    for conv_idx, layer in enumerate(all_conv_layers):
        try:
            save_path = os.path.join(neural_location,
                                     'natsound_activations',
                                     neural_analysis_string%layer.split('_fake')[0])
                        
            info = pickle.load(open(save_path, 'rb'))
            if init_conv:
                n_vox = info['r2s'].shape[0]
                n_samples = info['r2s'].shape[1]
                all_r2 = np.zeros([n_layers, n_vox, n_samples])*np.nan
                all_alphas = np.zeros([n_layers, n_vox, n_samples])*np.nan
                all_voxel_info = {'roi_masks': info['roi_masks'], 
                                  'voxel_meta': info['voxel_meta']}
                init_conv = False
            
            all_r2[conv_idx,:] = info['r2s']
            all_alphas[conv_idx,:] = info['alphas']
            
        except FileNotFoundError:
            print('%s not Found'%(os.path.join(neural_location,
                                               'natsound_activations',
                                               neural_analysis_string%layer)))
                        
    
    return all_r2, all_alphas, all_voxel_info, all_conv_layers

def responses_network_by_layer(all_participants_answers, experiment_params):
    experiment_response_dict = {}
    
    participants = list(all_participants_answers.keys())
    
    # Get the models that we will include in the plot
    models = experiment_params['experiment_info']['networks']
    experiment_response_dict = {model:{} for model in models}
    for p_idx, participant in enumerate(participants):
        for n_idx, network in enumerate(models):
            model_layers = all_model_info.ALL_NETWORKS_AND_LAYERS_AUDIO[network]['layers']
#             model_layers = [l for l in model_layers if ('input_after_preproc' not in l)]
            for l_idx, layer in enumerate(model_layers):
                if p_idx==0:
                    experiment_response_dict[network][layer]=[]
                saved_condition_name = '%d_%s/%d_%s'%(n_idx+1,network, l_idx, layer)
                experiment_response_dict[network][layer].append(
                    np.mean(all_participants_answers[participant][saved_condition_name]['correct_incorrect']))
                
    return experiment_response_dict

def combined_experiment_response_dictionaries(all_dicts):
    # all_dicts -- list of dictionaries output from respones_network_by_layer
    combined_experiment_dict = {}
    for experiment in all_dicts:
        for model in experiment.keys():            
            if model not in list(combined_experiment_dict.keys()):
                combined_experiment_dict[model] = experiment[model]
            else:
                print('Duplicate model %s'%model)
                for layer in combined_experiment_dict[model].keys():
                    combined_experiment_dict[model][layer] = combined_experiment_dict[model][layer] + experiment[model][layer]
                    
    return combined_experiment_dict

def choose_metamer_layer(metamer_responses_by_layer, measured_layer, return_type):
    all_human_responses = metamer_responses_by_layer[measured_layer] 
    if return_type=='average':
        average_metamer_recognition = np.mean(all_human_responses) # This value should vary between 0-1
        return average_metamer_recognition
    elif return_type=='sem':
        sem_metamer_recognition = np.std(all_human_responses) / np.sqrt(len(all_human_responses))
        return sem_metamer_recognition
    else:
        raise ValueError('Return type %s not recognized'%return_type)

def load_rsa_values(model_name, all_networks_and_layers, roi_name=None):
    all_conv_layers = all_networks_and_layers[model_name]['layers']

    rsa_base_path = '../assets/RSA_results/'
    if (roi_name is None) or (roi_name=='all_aud_voxels'):
        all_values = pickle.load(open(os.path.join(
            rsa_base_path, 'all_dataset_rsa_dict.pckl'), 'rb'))['NH2015']['trained'][model_name]
    else:
        all_values = pickle.load(open(os.path.join(
            rsa_base_path, 'all_dataset_roi_rsa_dict.pckl'), 'rb'))['NH2015'][roi_name]['trained'][model_name]
    return all_values, all_conv_layers

def plot_roi_analysis_for_models(models_to_plot,
                                 model_colors,
                                 model_style_dict,
                                 all_networks_and_layers,
                                 combined_experiment_dict,
                                 spectemp_layer='avgpool',
                                 roi_names = ['tonotopic', 'pitch', 'music', 'speech'],
                                 analysis_type='regression',
                                 ):
    """
    Makes the roi analysis neural predictions analysis plots.
    Holds one participant out and determines the "best layer" from the remaining participants. 
    Saves the subject-wise data as a .mat file for ANOVA analysis
    """
    if roi_names is None:
        plt.figure(figsize=(4,3))
    else:
        plt.figure(figsize=(3*len(roi_names),3))

    median_for_each_roi_and_model = {}
    models_roi_r2 = {}
    models_roi_alphas = {}
    
    all_model_results = []
    all_position_data = []
    all_metamer_best_layer = []
    all_metamer_best_layer_sem = []
    
    all_model_results_all_layers = []
    all_metamer_all_layers = []
    all_metamer_all_layers_sem = []
    
    bar_width = (0.7 / len(models_to_plot))
    bar_spacing = 0.05
    
    middle_idx = np.ceil(len(models_to_plot)/2)

    save_best_layer_names = {}
    for model_idx, model_name in enumerate(models_to_plot):
        save_best_layer_names[model_name] = {}
        # plt.subplot(1,len(models_to_plot),model_idx+1)
        if analysis_type=='regression':
            all_r2, all_alphas, all_voxel_info, all_conv_layers = load_all_voxel_predictions(all_networks_and_layers, model_name)
            if not all_voxel_info:
                print('### Missing all values for model %s ###'%model_name)
                all_model_results.append([np.ones(len(all_participants))*np.nan for roi_idx in range(len(roi_names))])
                all_position_data.append([np.ones(len(all_participants))*np.nan for roi_idx in range(len(roi_names))])
                all_metamer_best_layer.append([np.nan for roi_idx in range(len(roi_names))])
                all_metamer_best_layer_sem.append([np.nan for roi_idx in range(len(roi_names))])
                all_model_results_all_layers.append([np.ones(len(all_participants), len(all_conv_layers))*np.nan for roi_idx in range(len(roi_names))])
                continue
            roi_masks = all_voxel_info['roi_masks']
            
        elif analysis_type=='rsa':
            if model_idx==0:
                all_participants_noise_ceiling={}
        else:
            raise ValueError('Analyis type "%s" is not recognized'%analysis_type)
        
        
        
        all_roi_data = []
        all_roi_best_layer = []
        all_roi_data_all_layers = []
        num_voxels = {}
        for roi_idx, roi_name in enumerate(roi_names):  
            num_voxels[roi_name] = 0
            median_for_each_roi_and_model[model_name] = {}

            if analysis_type=='regression':
                participant_ids, all_participants = split_roi_data_by_participant(all_r2, 
                                                                                  all_voxel_info['voxel_meta'],
                                                                                  roi_masks)

                all_participants_median_layer_data = []
                all_participants_median_layer_data_mean_subtracted = []

                for participant_idx, participant_all_data in enumerate(all_participants):
                    participant = participant_all_data[roi_name]
                    # Take the mean across the 10 splits first, and use that as the layer-voxel pair. 
                    median_layer_voxel_data = np.nanmedian(participant,2)
                    # Take the median for each layer -- this is the value for each participant. 
                    median_layer_data = np.nanmedian(median_layer_voxel_data,1)

                    median_layer_data_mean_subtracted = median_layer_data - np.nanmean(median_layer_data)

                    all_participants_median_layer_data.append(median_layer_data)
                    all_participants_median_layer_data_mean_subtracted.append(median_layer_data_mean_subtracted)
                    
                    num_voxels[roi_name]+=median_layer_voxel_data.shape[1]

                all_participants_median_layer_data = np.array(all_participants_median_layer_data)

                participant_roi_from_held_out = []
                participant_best_layer_roi_from_held_out = []
                participant_all_layer_data = []
                for participant_idx, participant_all_data in enumerate(all_participants):
                    participants_to_measure = list(range(len(participant_ids)))
                    participants_to_measure.pop(participant_idx) # remove the participant we are holding out
                    # Kell2018 used a fisher transform here before averaging because these are r^2 values. 
                    # Not doing it here, but could if we want to fully replicate. 
                    average_layer_data = np.nanmean(all_participants_median_layer_data[participants_to_measure,:],0)
                    best_layer = np.argmax(average_layer_data)
                    participant_roi_from_held_out.append(np.nanmean(all_participants_median_layer_data[participant_idx, 
                                                                                                    best_layer]))
                    participant_best_layer_roi_from_held_out.append(best_layer)
                    
                    participant_all_layer_data.append(all_participants_median_layer_data[participant_idx,:])
            
            elif analysis_type=='rsa':
                all_values, all_conv_layers = load_rsa_values(model_name, all_networks_and_layers, roi_name=roi_name)
                participant_roi_from_held_out = []
                participant_best_layer_roi_from_held_out = []
                participant_all_layer_data = []
                if model_idx == 0:
                    all_participants_noise_ceiling[roi_name] = []
                all_participants = all_values.keys()
                
                for participant_idx, participant in enumerate(all_values.keys()):
                    # Take the median across splits -- this is the value for each participant. 
                    median_best_value = all_values[participant]['best_layer_rsa_median_across_splits']
                    participant_roi_from_held_out.append(median_best_value)
                    
                    best_layer = all_values[participant]['splits_median_best_layer_idx']

                    cross_val_neural = all_values[participant]['leave_one_out_neural_r_cv_test_splits']
                    if model_idx==0:
                        all_participants_noise_ceiling[roi_name].append(np.median(cross_val_neural))
                        
                    participant_best_layer_roi_from_held_out.append(best_layer)
                    
                    participant_all_layer_data.append(all_values[participant]['splits_median_distance_all_layers'])
            
            all_roi_data.append(participant_roi_from_held_out)
            all_roi_best_layer.append(participant_best_layer_roi_from_held_out)
            
            all_roi_data_all_layers.append(participant_all_layer_data)
        
        all_roi_best_layer = np.array(all_roi_best_layer)
        median_best_layer = scipy.stats.mode(all_roi_best_layer, axis=1).mode
        median_best_layer_name = [all_conv_layers[np.squeeze(int(m))] for m in median_best_layer]
        save_best_layer_names[model_name] = {roi_name:median_best_layer_name[roi_idx] for roi_idx, roi_name in enumerate(roi_names)}
        
        average_metamer_recognition = [choose_metamer_layer(combined_experiment_dict[model_name], m, 'average') for m in median_best_layer_name]
        sem_metamer_recognition = [choose_metamer_layer(combined_experiment_dict[model_name], m, 'sem') for m in median_best_layer_name]
        all_metamer_best_layer.append(average_metamer_recognition)
        all_metamer_best_layer_sem.append(sem_metamer_recognition)
        
        # Get metamer recognition for all layers
        average_all_metamer_recognition = [choose_metamer_layer(combined_experiment_dict[model_name], m, 'average') for m in all_conv_layers]
        sem_all_metamer_recognition = [choose_metamer_layer(combined_experiment_dict[model_name], m, 'sem') for m in all_conv_layers]

        all_roi_data = np.array(all_roi_data)
        mean_subtract_roi_data = all_roi_data - np.mean(all_roi_data,0)
        
        sem = np.std(all_roi_data, 1)/np.sqrt(len(all_participants))
        
        roi_mean = np.mean(all_roi_data, 1)
        
        all_model_results.append(all_roi_data)
        
        all_model_results_all_layers.append(all_roi_data_all_layers)
        all_metamer_all_layers.append(average_all_metamer_recognition)
        all_metamer_all_layers_sem.append(sem_all_metamer_recognition)
                    
    num_models, num_rois, num_participants = np.array(all_model_results).shape    
    
    # Scatter of predictivity vs. metamer recognition, best layer
    all_metamer_best_layer = np.array(all_metamer_best_layer)
    all_metamer_best_layer_sem = np.array(all_metamer_best_layer_sem)
    plt.figure(figsize=(4*num_rois,4))
    
    for roi_plot_idx, roi_name in enumerate(roi_names):
        all_scores_list = []
        all_metamers_list = []
        
        plt.subplot(1,num_rois,roi_plot_idx+1)
        for model_idx, model_name in enumerate(models_to_plot):
            score_val = np.array(all_model_results)[model_idx, roi_plot_idx, :].mean(axis=0)
            score_sem = (np.array(all_model_results)[model_idx, roi_plot_idx, :].std(axis=0))/np.sqrt(num_participants)
            metamer_val = all_metamer_best_layer[model_idx,roi_plot_idx]
            all_scores_list.append(score_val)
            all_metamers_list.append(metamer_val)
            markers, caps, bars = plt.errorbar(
                         score_val,
                         metamer_val,
                         yerr=all_metamer_best_layer_sem[model_idx,roi_plot_idx], 
                         xerr=score_sem, 
                         fmt=model_style_dict[model_name],
                         color=model_colors[model_name],
                         label=model_name,
                         linestyle='',
                         ms=8,
                         )
            [bar.set_alpha(0.4) for bar in bars]
            [cap.set_alpha(0.4) for cap in caps]
            
            
        if analysis_type=='rsa':
            mean_noise_ceiling = np.mean(all_participants_noise_ceiling[roi_name])
            sem_noise_ceiling = np.std(all_participants_noise_ceiling[roi_name])/np.sqrt(num_participants)

            plt.plot([mean_noise_ceiling,mean_noise_ceiling], 
                     [0, 1],'k--',
                     solid_capstyle='butt',
                    alpha=1, label='_nolegend_'
                   )

            plt.fill_betweenx([0, 1], 
                             mean_noise_ceiling-sem_noise_ceiling, 
                             mean_noise_ceiling+sem_noise_ceiling,
                             facecolor='k', alpha=0.5, label='_nolegend_'
                            )
            
        # For the evaluated  correlations, we Bonferroni corrected the p-values 
        # for the correlation between model metamer recognizability and variance 
        # explained by multiplying the p-value by 5 (the number of tests performed).
        bonferroni_correction_factor = 5
        rho_val, p_val_rho_raw = scipy.stats.spearmanr(np.array(all_scores_list), np.array(all_metamers_list))
        p_val_rho = np.min([p_val_rho_raw * bonferroni_correction_factor, 1.]) # Max out at 1. 
        
        r_val, p_val_r_raw = scipy.stats.pearsonr(np.array(all_scores_list), np.array(all_metamers_list))
        p_val_r = np.min([p_val_r_raw * bonferroni_correction_factor, 1.])
        
        plt.text(0.01,0.25, 'rho=%0.3f, p=%0.2f, (raw p=%0.2f)'%(rho_val, p_val_rho, p_val_rho_raw))
        plt.text(0.01,0.15, 'R^2=%0.3f, p=%0.2f, (raw p=%0.2f)'%(r_val**2, p_val_r, p_val_r_raw))
        
        if roi_plot_idx == num_rois-1:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, ncol=2)
        plt.title(roi_names[roi_plot_idx])
        plt.ylabel('Human recognition of model\nmetamers from best model stage')
        if analysis_type == 'regression':
            plt.xlabel('Variance Explained fMRI (R^2)')
        elif analysis_type == 'rsa':
            plt.xlabel('RSA Value')
        plt.xlim([0,1])
        plt.ylim([0,1])
        
#     plt.savefig('2023_07_09_AuditoryfMRI_%s_BestLayer.pdf'%(analysis_type))
        
    return save_best_layer_names


