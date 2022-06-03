%% Run this files to compute the network-human main effect and interactions for all for the VISION experiments
%% Must specify the paths to the required network data mat files. 

function [] = run_anova_stats_matlab_network_human_results(EXP_NUM)
rng(517)
addpath('../matlab_statistics_functions')

NUM_BOOTSTRAPS = 10000

if EXP_NUM==1
%%
experiment_name = 'Auditory Networks: Standard Supervised Experiment';
experiment_analysis_path = 'EXPERIMENT_1';
model_names_and_files = {...
    'EXP1 - CochResNet50 (Standard)', 'EXPERIMENT_1/AudioExperiment1_network_vs_humans_datamatrix_cochresnet50.mat'; ...
    'EXP1 - CochCNN9 (Standard)', 'EXPERIMENT_1/AudioExperiment1_network_vs_humans_datamatrix_kell2018.mat'; ...
};
save_all_experiment_anovas(experiment_name, experiment_analysis_path, model_names_and_files, NUM_BOOTSTRAPS)

elseif EXP_NUM==3
%% 
experiment_name = 'Auditory Networks: Waveform Adversarially Trained CochResNet50 Experiment';
experiment_analysis_path = 'EXPERIMENT_3';
model_names_and_files = {...
    'EXP3 - CochResNet50 (Standard)', 'EXPERIMENT_3/AudioExperiment3_network_vs_humans_datamatrix_cochresnet50.mat'; ...
    'EXP3 - CochResNet50 Wav Adversarial L2 Norm (EPS=1)', 'EXPERIMENT_3/AudioExperiment3_network_vs_humans_datamatrix_cochresnet50_l2_1_robust_waveform.mat'; ...
    'EXP3 - CochResNet50 Wav Adversarial L2 Norm (EPS=0.5)', 'EXPERIMENT_3/AudioExperiment3_network_vs_humans_datamatrix_cochresnet50_l2_p5_robust_waveform.mat'; ...
    'EXP3 - CochResNet50 Wav Random L2 Norm (EPS=1)', 'EXPERIMENT_3/AudioExperiment3_network_vs_humans_datamatrix_cochresnet50_l2_1_random_step_waveform.mat'; ...
    'EXP3 - CochResNet50 Wav Adversarial Linf Norm (EPS=0.002)', 'EXPERIMENT_3/AudioExperiment3_network_vs_humans_datamatrix_cochresnet50_linf_p002_robust_waveform.mat'; ...
    'EXP3 - CochResNet50 Wav Random Linf Norm (EPS=0.002)', 'EXPERIMENT_3/AudioExperiment3_network_vs_humans_datamatrix_cochresnet50_linf_p002_random_step_waveform.mat'; ...
    };
save_all_experiment_anovas(experiment_name, experiment_analysis_path, model_names_and_files, NUM_BOOTSTRAPS)

elseif EXP_NUM==4
%%
experiment_name = 'Auditory Networks: Waveform Adversarially Trained CochCNN9 Experiment';
experiment_analysis_path = 'EXPERIMENT_4';
model_names_and_files = {...
    'EXP4 - CochCNN9 (Standard)', 'EXPERIMENT_4/AudioExperiment4_network_vs_humans_datamatrix_kell2018.mat'; ...
    'EXP4 - CochCNN9 Wav Adversarial L2 Norm (EPS=1)', 'EXPERIMENT_4/AudioExperiment4_network_vs_humans_datamatrix_kell2018_l2_1_robust_waveform.mat'; ...
    'EXP4 - CochCNN9 Wav Random L2 Norm (EPS=1)', 'EXPERIMENT_4/AudioExperiment4_network_vs_humans_datamatrix_kell2018_l2_1_random_step_waveform.mat'; ...
    'EXP4 - CochCNN9 Wav Adversarial Linf Norm (EPS=0.002)', 'EXPERIMENT_4/AudioExperiment4_network_vs_humans_datamatrix_kell2018_linf_p002_robust_waveform.mat'; ...
    'EXP4 - CochCNN9 Wav Random Linf Norm (EPS=0.002)', 'EXPERIMENT_4/AudioExperiment4_network_vs_humans_datamatrix_kell2018_linf_p002_random_step_waveform.mat'; ...
    };
save_all_experiment_anovas(experiment_name, experiment_analysis_path, model_names_and_files, NUM_BOOTSTRAPS)

elseif EXP_NUM==6
%%
experiment_name = 'Audio Networks: Classical Models, SpecTemp';
experiment_analysis_path = 'EXPERIMENT_6';
model_names_and_files = { ...
    'EXP6 - SpecTemp', 'EXPERIMENT_6/AudioExperiment6_network_vs_humans_datamatrix_spectemp_filters.mat'; ...
};
save_all_experiment_anovas(experiment_name, experiment_analysis_path, model_names_and_files, NUM_BOOTSTRAPS)


end
end

%% Helper functions for running all of the models through and saving the results

function [] = save_all_experiment_anovas(experiment_name, experiment_analysis_path, model_names_and_files, NUM_BOOTSTRAPS)
save_matlab_output_path = [experiment_analysis_path, '/', date, '-', experiment_analysis_path, '_ANOVA_NETWORK_HUMAN_RESULTS.txt']

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
          p_value_main_effect, ...
          F_stage_observer_interaction, ...
          p_value_interaction] = run_network_human_anova_from_data_matrix_path(model_name, ...
                                                                               data_matrix_path, ...
                                                                               NUM_BOOTSTRAPS);

    end
end

