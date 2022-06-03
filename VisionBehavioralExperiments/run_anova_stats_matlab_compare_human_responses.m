%% Run this files to compute the network-human main effect and interactions for all for the VISION experiments
%% Must specify the paths to the required network data mat files. 

function run_anova_stats_matlab_compare_human_responses(EXP_NUM, NUM_BOOTSTRAPS)
rng(517)
addpath('../matlab_statistics_functions')

%% NO ANALYSIS FOR EXPERIMENT 1
%  experiment_name = 'Visual Networks: Standard Supervised Experiment';
% experiment_analysis_path = 'EXP1_ANALYSIS';

if EXP_NUM==3
%% EXPERIMENT 3 
experiment_name = 'Visual Networks: Robust ResNet50 Experiment';
experiment_analysis_path = 'EXP3_ANALYSIS';
experiment_data_matrix_file = 'EXP3_ANALYSIS/VisionExperiment3_datamatrix.mat'
% Run a comparison for each figure panel in Figure 7 -- ResNet50
% Adversarial vs. Standard Tests
anova_test_name = 'l2_adversarial_vs_standard_resnet50'
include_models = {'resnet50', 'resnet50_l2_3_robust'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ... 
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'linf4_adversarial_vs_standard_resnet50'
include_models = {'resnet50', 'resnet50_linf_4_robust'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'linf8_adversarial_vs_standard_resnet50'
include_models = {'resnet50', 'resnet50_linf_8_robust'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Random vs Standard Tests
anova_test_name = 'l2_random_vs_standard_resnet50'
include_models = {'resnet50', 'resnet50_random_l2_perturb'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'linf_random_vs_standard_resnet50'
include_models = {'resnet50', 'resnet50_random_linf8_perturb'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Adversarial vs Random Tests
anova_test_name = 'l2_adversarial_vs_l2_random_resnet50'
include_models = {'resnet50_l2_3_robust', 'resnet50_random_l2_perturb'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'linf8_adversarial_vs_linf8_random'
include_models = {'resnet50_linf_8_robust', 'resnet50_random_linf8_perturb'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Types of adversaries
anova_test_name = 'linf_adversarial_vs_l2_adversarial'
include_models = {'resnet50_linf_4_robust', 'resnet50_linf_8_robust', 'resnet50_l2_3_robust'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)
elseif EXP_NUM==4
%% EXPERIMENT 4
experiment_name = 'Visual Networks: Robust AlexNet Experiment';
experiment_analysis_path = 'EXP4_ANALYSIS';
experiment_data_matrix_file = 'EXP4_ANALYSIS/VisionExperiment4_datamatrix.mat'
% Run a comparison for each figure panel in Figure 7 -- AlexNet
% Adversarial vs. Standard Tests
anova_test_name = 'l2_adversarial_vs_standard_alexnet'
include_models = {'alexnet', 'alexnet_l2_3_robust'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'linf_adversarial_vs_standard_alexnet'
include_models = {'alexnet', 'alexnet_linf_8_robust'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Random vs Standard Tests
anova_test_name = 'l2_random_vs_standard_alexnet'
include_models = {'alexnet', 'alexnet_random_l2_3_perturb'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'linf_random_vs_standard_alexnet'
include_models = {'alexnet', 'alexnet_random_linf8_perturb'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Adversarial vs Random Tests
anova_test_name = 'l2_adversarial_vs_l2_random_alexnet'
include_models = {'alexnet_l2_3_robust', 'alexnet_random_l2_3_perturb'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

anova_test_name = 'linf_adversarial_vs_linf_random'
include_models = {'alexnet_linf_8_robust', 'alexnet_random_linf8_perturb'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

% Types of adversaries
anova_test_name = 'linf_adversarial_vs_l2_adversarial'
include_models = {'alexnet_linf_8_robust', 'alexnet_l2_3_robust'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)

elseif EXP_NUM==5
%%
experiment_name = 'Visual Networks: Self-Supervised Experiment';
experiment_analysis_path = 'EXP5_ANALYSIS';
experiment_data_matrix_file = 'EXP5_ANALYSIS/VisionExperiment5_datamatrix.mat'
% Run a comparison between any of the network types?
% Or should we run one comparison for each network type again standard?
anova_test_name = 'simclr_mocov2_byol_and_standard'
include_models = {'resnet50', 'resnet50_simclr', 'resnet50_moco_v2', 'resnet50_byol'}
save_experiment_anova_compare_human_responses(experiment_name, experiment_analysis_path, ...
                                              experiment_data_matrix_file, anova_test_name, ...
                                              include_models, NUM_BOOTSTRAPS)


elseif EXP_NUM==9
%% 
experiment_name = 'Visual Networks: GVOneNet and LowpassAlexNet Experiment';
experiment_analysis_path = 'EXP9_ANALYSIS';
experiment_data_matrix_file = 'EXP9_ANALYSIS/VisionExperiment9_datamatrix.mat'
% Compare GVOneNEt and LowPassAlexnet
anova_test_name = 'GVOneNet_LowPassAlexnet'
include_models = {'alexnet_reduced_aliasing_early_checkpoint', 'vonealexnet_gaussian_noise_std4_fixed'}
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

