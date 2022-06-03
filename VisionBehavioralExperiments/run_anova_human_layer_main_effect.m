%% Run this files to compute the network-human main effect and interactions for all for the VISION experiments
%% Must specify the paths to the required network data mat files. 

function [] = run_anova_human_layer_main_effect(EXP_NUM)
rng(517)
addpath('../matlab_statistics_functions')

NUM_BOOTSTRAPS = 10000

if EXP_NUM==1
%%
experiment_name = 'Visual Networks: Standard Supervised Experiment';
experiment_analysis_path = 'EXP1_ANALYSIS';
model_names_and_files = {'EXP1 - CORnet-S (Standard)', 'EXP1_ANALYSIS/VisionExperiment1_network_vs_humans_datamatrix_cornet_s.mat'; ...
    'EXP1 - VGG-19 (Standard)', 'EXP1_ANALYSIS/VisionExperiment1_network_vs_humans_datamatrix_vgg_19.mat'; ...
    'EXP1 - ResNet50 (Standard)', 'EXP1_ANALYSIS/VisionExperiment1_network_vs_humans_datamatrix_resnet50.mat'; ...
    'EXP1 - ResNet101 (Standard)', 'EXP1_ANALYSIS/VisionExperiment1_network_vs_humans_datamatrix_resnet101.mat'; ...
    'EXP1 - AlexNet (Standard)', 'EXP1_ANALYSIS/VisionExperiment1_network_vs_humans_datamatrix_alexnet.mat'; ...
};
save_all_experiment_anovas(experiment_name, experiment_analysis_path, model_names_and_files, NUM_BOOTSTRAPS)

elseif EXP_NUM==3
%% 
experiment_name = 'Visual Networks: Robust ResNet50 Experiment';
experiment_analysis_path = 'EXP3_ANALYSIS';
model_names_and_files = {'EXP3 - ResNet50 (Standard)', 'EXP3_ANALYSIS/VisionExperiment3_network_vs_humans_datamatrix_resnet50.mat'; ...
    'EXP3 - ResNet50 Adversarial L2 Norm (EPS=3)', 'EXP3_ANALYSIS/VisionExperiment3_network_vs_humans_datamatrix_resnet50_l2_3_robust.mat'; ...
    'EXP3 - ResNet50 Random L2 Norm (EPS=3)', 'EXP3_ANALYSIS/VisionExperiment3_network_vs_humans_datamatrix_resnet50_random_l2_perturb.mat'; ...
    'EXP3 - ResNet50 Adversarial Linf Norm (EPS=4)', 'EXP3_ANALYSIS/VisionExperiment3_network_vs_humans_datamatrix_resnet50_linf_4_robust.mat'; ...
    'EXP3 - ResNet50 Adversarial Linf Norm (EPS=8)', 'EXP3_ANALYSIS/VisionExperiment3_network_vs_humans_datamatrix_resnet50_linf_8_robust.mat'; ...
    'EXP3 - ResNet50 Random Linf Norm (EPS=8)', 'EXP3_ANALYSIS/VisionExperiment3_network_vs_humans_datamatrix_resnet50_random_linf8_perturb.mat'; ...
    };
save_all_experiment_anovas(experiment_name, experiment_analysis_path, model_names_and_files, NUM_BOOTSTRAPS)

elseif EXP_NUM==4
%%
experiment_name = 'Visual Networks: Robust AlexNet Experiment';
experiment_analysis_path = 'EXP4_ANALYSIS';
model_names_and_files = {'EXP4 - AlexNet (Standard)', 'EXP4_ANALYSIS/VisionExperiment4_network_vs_humans_datamatrix_alexnet.mat'; ...
     'EXPi - AlexNet Adversarial L2 Norm (EPS=3)', 'EXP4_ANALYSIS/VisionExperiment4_network_vs_humans_datamatrix_alexnet_l2_3_robust.mat'; ...
    'EXP4 - AlexNet Random L2 Norm (EPS=3)', 'EXP4_ANALYSIS/VisionExperiment4_network_vs_humans_datamatrix_alexnet_random_l2_3_perturb.mat';...
    'EXP4 - AlexNet Adversarial Linf Norm (EPS=8)', 'EXP4_ANALYSIS/VisionExperiment4_network_vs_humans_datamatrix_alexnet_linf_8_robust.mat'; ...
    'EXP4 - AlexMet Random Linf Norm (EPS=8)', 'EXP4_ANALYSIS/VisionExperiment4_network_vs_humans_datamatrix_alexnet_random_linf8_perturb.mat'; ...
    };
save_all_experiment_anovas(experiment_name, experiment_analysis_path, model_names_and_files, NUM_BOOTSTRAPS)

elseif EXP_NUM==5
%%
experiment_name = 'Visual Networks: Self-Supervised Experiment';
experiment_analysis_path = 'EXP5_ANALYSIS';
model_names_and_files = { ...
    'EXP5 - ResNet50 (Standard)', 'EXP5_ANALYSIS/VisionExperiment5_network_vs_humans_datamatrix_resnet50.mat'; ...
    'EXP5 - ResNet50 SIMCLR (Self-Supervised)', 'EXP5_ANALYSIS/VisionExperiment5_network_vs_humans_datamatrix_resnet50_simclr.mat'; ...
    'EXP5 - ResNet50 MOCO_V2 (Self-Supervised)', 'EXP5_ANALYSIS/VisionExperiment5_network_vs_humans_datamatrix_resnet50_moco_v2.mat'; ...
    'EXP5 - ResNet50 BYOL (Self-Supervised)', 'EXP5_ANALYSIS/VisionExperiment5_network_vs_humans_datamatrix_resnet50_byol.mat'; ...
};
save_all_experiment_anovas(experiment_name, experiment_analysis_path, model_names_and_files, NUM_BOOTSTRAPS)

elseif EXP_NUM==6
%%
experiment_name = 'Visual Networks: Classical Models, HMAX';
experiment_analysis_path = 'EXP6_ANALYSIS';
model_names_and_files = { ...
    'EXP6 - HMAX', 'EXP6_ANALYSIS/VisionExperiment6_network_vs_humans_datamatrix_hmax_standard.mat'; ...
};
save_all_experiment_anovas(experiment_name, experiment_analysis_path, model_names_and_files, NUM_BOOTSTRAPS)

elseif EXP_NUM==9
%% 
experiment_name = 'Visual Networks: GVOneNet and LowpassAlexNet Experiment';
experiment_analysis_path = 'EXP9_ANALYSIS';
model_names_and_files = {'EXP9 - AlexNet, Early Checkpoint (Standard)', 'EXP9_ANALYSIS/VisionExperiment9_network_vs_humans_datamatrix_alexnet_early_checkpoint.mat'; ...
    'EXP9 - LowpassAlexNet', 'EXP9_ANALYSIS/VisionExperiment9_network_vs_humans_datamatrix_alexnet_reduced_aliasing_early_checkpoint.mat'; ...
    'EXP9 - GVOneAlexNet', 'EXP9_ANALYSIS/VisionExperiment9_network_vs_humans_datamatrix_vonealexnet_gaussian_noise_std4_fixed.mat'; ...
    };

save_all_experiment_anovas(experiment_name, experiment_analysis_path, model_names_and_files, NUM_BOOTSTRAPS)
end
end

%% Helper functions for running all of the models through and saving the results

function [] = save_all_experiment_anovas(experiment_name, experiment_analysis_path, model_names_and_files, NUM_BOOTSTRAPS)
save_matlab_output_path = [experiment_analysis_path, '/', date, '-', experiment_analysis_path, '_MAIN_EFFECT_LAYER_HUMAN_RESULTS.txt']

results = evalc('run_experiment(experiment_name, model_names_and_files, NUM_BOOTSTRAPS)')
remove_bold_results = regexprep(results, {'<strong>', '</strong>'}, {'', ''});

fid = fopen(save_matlab_output_path,'w');
fprintf(fid,'%s',remove_bold_results);
fclose(fid);
end

function [] = run_experiment(experiment_name, model_names_and_files, NUM_BOOTSTRAPS)
    disp(['######### | ' experiment_name ' | #########'])
    disp(['NUM_PERMUTATIONS = ' num2str(NUM_BOOTSTRAPS)])
    for model_idx=1:size(model_names_and_files,1)
         model_name = model_names_and_files{model_idx, 1};
         data_matrix_path = model_names_and_files{model_idx, 2};

         [F_observer_main_effect, ...
          p_value_main_effect] = run_human_layer_effect_anova_from_data_matrix_path(model_name, ...
                                                                               data_matrix_path, ...
                                                                               NUM_BOOTSTRAPS);

    end
end

