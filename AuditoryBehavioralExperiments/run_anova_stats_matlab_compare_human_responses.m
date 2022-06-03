%% Run this files to compute the network-human main effect and interactions for all for the VISION experiments
%% Must specify the paths to the required network data mat files. 

function run_anova_stats_matlab_compare_human_responses(EXP_NUM, NUM_BOOTSTRAPS)
rng(517)

addpath('../matlab_statistics_functions')

%% NO ANALYSIS FOR EXPERIMENT 1
%  experiment_name = 'Auditory Networks: Standard Supervised Experiment';
% experiment_analysis_path = 'EXP1_ANALYSIS';

if EXP_NUM==3
%% EXPERIMENT 3 
experiment_name = 'Auditory Networks: Robust CochResNet50 Experiment, Waveform Adversaries';
experiment_analysis_path = 'EXPERIMENT_3';
experiment_data_matrix_file = 'EXPERIMENT_3/AudioExperiment3_datamatrix.mat'
% Run a comparison for each figure panel in Figure 7 -- ResNet50
% Adversarial vs. Standard Tests
anova_test_name = 'l2_1_adversarial_vs_standard_cochresnet50_waveform'
include_models = {'cochresnet50', 'cochresnet50_l2_1_robust_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'l2_p5_adversarial_vs_standard_cochresnet50_waveform'
include_models = {'cochresnet50', 'cochresnet50_l2_p5_robust_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'linf_adversarial_vs_standard_cochresnet50_waveform'
include_models = {'cochresnet50', 'cochresnet50_linf_p002_robust_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Random vs Standard Tests
anova_test_name = 'l2_random_vs_standard_cochresnet50_waveform'
include_models = {'cochresnet50', 'cochresnet50_l2_1_random_step_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'linf_random_vs_standard_cochresnet50_waveform'
include_models = {'cochresnet50', 'cochresnet50_linf_p002_random_step_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Adversarial vs Random Tests
anova_test_name = 'l2_1_adversarial_vs_l2_1_random_cochresnet50_waveform'
include_models = {'cochresnet50_l2_1_robust_waveform', 'cochresnet50_l2_1_random_step_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'linf_adversarial_vs_linf_random_cochresnet50_waveform'
include_models = {'cochresnet50_linf_p002_robust_waveform', 'cochresnet50_linf_p002_random_step_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Types of adversaries
anova_test_name = 'linf_adversarial_vs_l2_adversarial'
include_models = {'cochresnet50_linf_p002_robust_waveform', 'cochresnet50_l2_1_robust_waveform', 'cochresnet50_l2_p5_robust_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)
elseif EXP_NUM==4
%% EXPERIMENT 4
experiment_name = 'Auditory Networks: Robust CochCNN9 (kel2018) Experiment';
experiment_analysis_path = 'EXPERIMENT_4';
experiment_data_matrix_file = 'EXPERIMENT_4/AudioExperiment4_datamatrix.mat'
% Run a comparison for each figure panel in Figure 7 -- CochCNN9
% Adversarial vs. Standard Tests
anova_test_name = 'l2_adversarial_vs_standard_kell2018_waveform'
include_models = {'kell2018', 'kell2018_l2_1_robust_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'linf_adversarial_vs_standard_kell2018_waveform'
include_models = {'kell2018', 'kell2018_linf_p002_robust_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Random vs Standard Tests
anova_test_name = 'l2_random_vs_standard_kell2018_waveform'
include_models = {'kell2018', 'kell2018_l2_1_random_step_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'linf_random_vs_standard_kell2018_waveform'
include_models = {'kell2018', 'kell2018_linf_p002_random_step_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Adversarial vs Random Tests
anova_test_name = 'l2_adversarial_vs_l2_random_kell2018_waveform'
include_models = {'kell2018_l2_1_robust_waveform', 'kell2018_l2_1_random_step_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'linf_adversarial_vs_linf_random_kell2018_waveform'
include_models = {'kell2018_linf_p002_robust_waveform', 'kell2018_linf_p002_random_step_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Types of adversaries
anova_test_name = 'linf_adversarial_vs_l2_adversarial_kell2018_waveform'
include_models = {'kell2018_l2_1_robust_waveform', 'kell2018_linf_p002_robust_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

elseif EXP_NUM==7
%% EXPERIMENT 7
experiment_name = 'Auditory Networks: Robust CochResNet50 Experiment, Cochleagram Adversaries';
experiment_analysis_path = 'EXPERIMENT_7';
experiment_data_matrix_file = 'EXPERIMENT_7/AudioExperiment7_datamatrix.mat'
% Adversarial vs. Standard Tests
anova_test_name = 'l2_1_adversarial_coch_vs_standard_cochresnet50_cochleagram'
include_models = {'cochresnet50', 'cochresnet50_l2_1_robust_cochleagram'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'l2_p5_adversarial_coch_vs_standard_cochresnet50_cochleagram'
include_models = {'cochresnet50', 'cochresnet50_l2_p5_robust_cochleagram'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Random vs Standard Tests
anova_test_name = 'l2_random_vs_standard_cochresnet50_cochleagram'
include_models = {'cochresnet50', 'cochresnet50_l2_1_random_step_cochleagram'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Adversarial vs Random Tests
anova_test_name = 'l2_1_adversarial_coch_vs_l2_random_cochresnet50_cochleagram'
include_models = {'cochresnet50_l2_1_robust_cochleagram', 'cochresnet50_l2_1_random_step_cochleagram'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'l2_p5_adversarial_coch_vs_l2_random_cochresnet50_cochleagram'
include_models = {'cochresnet50_l2_p5_robust_cochleagram', 'cochresnet50_l2_1_random_step_cochleagram'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Waveform vs Cochleagram Tests
anova_test_name = 'l2_1_adversarial_waveform_vs_l2_1_adversarial_cochleagram_cochresnet50'
include_models = {'cochresnet50_l2_1_robust_cochleagram', 'cochresnet50_l2_1_robust_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'l2_1_adversarial_waveform_vs_l2_p5_adversarial_cochleagram_cochresnet50'
include_models = {'cochresnet50_l2_p5_robust_cochleagram', 'cochresnet50_l2_1_robust_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Perturbation size test
anova_test_name = 'l2_adversarial_cochleagram_perturbation_size_cochresnet50'
include_models = {'cochresnet50_l2_1_robust_cochleagram', 'cochresnet50_l2_p5_robust_cochleagram'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

elseif EXP_NUM==8
%% EXPERIMENT 8
experiment_name = 'Auditory Networks: Robust CochCNN9 (kell2018) Experiment, Cochleagram Adversaries';
experiment_analysis_path = 'EXPERIMENT_8';
experiment_data_matrix_file = 'EXPERIMENT_8/AudioExperiment8_datamatrix.mat'
% Adversarial vs. Standard Tests
anova_test_name = 'l2_1_adversarial_vs_standard_kell2018_cochleagram'
include_models = {'kell2018', 'kell2018_l2_1_robust_cochleagram'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'l2_p5_adversarial_vs_standard_kell2018_cochleagram'
include_models = {'kell2018', 'kell2018_l2_p5_robust_cochleagram'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Random vs Standard Tests
anova_test_name = 'l2_random_vs_standard_kell2018_cochleagram'
include_models = {'kell2018', 'kell2018_l2_1_random_step_cochleagram'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Adversarial vs Random Tests
anova_test_name = 'l2_1_adversarial_vs_l2_1_random_kell2018_cochleagram'
include_models = {'kell2018_l2_1_robust_cochleagram', 'kell2018_l2_1_random_step_cochleagram'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Waveform vs Cochleagram Tests
anova_test_name = 'l2_1_adversarial_waveform_vs_l2_1_adversarial_cochleagram_kell2018'
include_models = {'kell2018_l2_1_robust_cochleagram', 'kell2018_l2_1_robust_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'l2_1_adversarial_waveform_vs_l2_p5_adversarial_cochleagram_kell2018'
include_models = {'kell2018_l2_p5_robust_cochleagram', 'kell2018_l2_1_robust_waveform'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Perturbation size test
anova_test_name = 'l2_adversarial_cochleagram_perturbation_size_kell2018'
include_models = {'kell2018_l2_1_robust_cochleagram', 'kell2018_l2_p5_robust_cochleagram'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)


end
end


%% Helper functions for running all of the models through and saving the results
function [] = save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)
stats_folder = ['STATS-' date]
mkdir(experiment_analysis_path, stats_folder)
save_matlab_output_path = [experiment_analysis_path, '/', stats_folder, '/', experiment_analysis_path, '-', ...
                           anova_test_name, '-N_', num2str(NUM_BOOTSTRAPS), '-', 'ANOVA_HUMAN_RESULTS.txt']

results = evalc('run_experiment(experiment_name, anova_test_name, experiment_data_matrix_file, include_models, NUM_BOOTSTRAPS)')
remove_bold_results = regexprep(results, {'<strong>', '</strong>'}, {'', ''});

save_matlab_output_path 
fid = fopen(save_matlab_output_path,'w');
fprintf(fid,'%s',remove_bold_results);
fclose(fid);
end

function [] = run_experiment(experiment_name, anova_test_name, experiment_data_matrix_file, ...
                             include_models, NUM_BOOTSTRAPS)
    disp(['######### | ' experiment_name ' - ' anova_test_name ' | #########'])
    disp(['NUM_PERMUTATIONS = ' num2str(NUM_BOOTSTRAPS)])
    disp(['MODELS_INCLUDED_IN_ANOVA: ' strjoin(include_models, ', ')])
    load(experiment_data_matrix_file)
    model_idx = arrayfun(@(t)(strmatch(t, models, 'exact')), include_models)
    within_factor_names = {'layer', 'model_type'};
    anova_comparison_data_matrix = participant_data_matrix(:,:,model_idx)

    true_values_statistics = simple_mixed_anova_partialeta(anova_comparison_data_matrix, [], within_factor_names);
    
    [p_value_interaction, p_value_main_effect_model] = run_permutation_network_types(...
        true_values_statistics, anova_comparison_data_matrix, ...
        within_factor_names, ...
        NUM_BOOTSTRAPS ...
        )

    disp([anova_test_name ' Full ANOVA'])
    disp(true_values_statistics)

    disp(['F(model) main effect: ' num2str(true_values_statistics{'(Intercept):model_type','F'})])
    disp(['p_value main effect : ' num2str(p_value_main_effect_model)])

    disp(['F(model, stage) interaction: ' num2str(true_values_statistics{'(Intercept):layer:model_type','F'})])
    disp(['p_value interaction : ' num2str(p_value_interaction)])
end

