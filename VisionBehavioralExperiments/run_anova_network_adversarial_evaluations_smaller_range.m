%% Run this files to compute the network-human main effect and interactions for all for the VISION experiments
%% Must specify the paths to the required network data mat files. 

function [] = run_anova_stats_matlab_network_human_results(EXP_NUM)
rng(517)
addpath('../matlab_statistics_functions')

NUM_BOOTSTRAPS = 10000

if EXP_NUM==9
%% 
experiment_name = 'Visual Networks: GVOneNet and LowpassAlexNet Experiment';
experiment_analysis_path = 'EXP9_ANALYSIS';
analysis_short_name_full_names_files_and_models = { ...
    'gvone_vs_lowpass_alexnet_adv_eval_l1', 'EXP9 - L1: VOneAlexNet (std=4) vs. LowPassAlexNet', 'EXP9_ANALYSIS/l1_alexnet_early_checkpoint-alexnet_reduced_aliasing_early_checkpoint-vonealexnet_gaussian_noise_std4_fixed.mat', {'alexnet_reduced_aliasing_early_checkpoint', 'vonealexnet_gaussian_noise_std4_fixed'}; ...
    'gvone_vs_lowpass_alexnet_adv_eval_l2', 'EXP9 - L2: VOneAlexNet (std=4) vs. LowPassAlexNet', 'EXP9_ANALYSIS/l2_alexnet_early_checkpoint-alexnet_reduced_aliasing_early_checkpoint-vonealexnet_gaussian_noise_std4_fixed.mat', {'alexnet_reduced_aliasing_early_checkpoint', 'vonealexnet_gaussian_noise_std4_fixed'}; ...
    'gvone_vs_lowpass_alexnet_adv_eval_linf', 'EXP9 - Linf: VOneAlexNet (std=4) vs. LowPassAlexNet', 'EXP9_ANALYSIS/linf_alexnet_early_checkpoint-alexnet_reduced_aliasing_early_checkpoint-vonealexnet_gaussian_noise_std4_fixed.mat', {'alexnet_reduced_aliasing_early_checkpoint', 'vonealexnet_gaussian_noise_std4_fixed'}; ...
    'gvone_vs_standard_alexnet_adv_eval_l1', 'EXP9 - L1: VOneAlexNet (std=4) vs. Standard', 'EXP9_ANALYSIS/l1_alexnet_early_checkpoint-alexnet_reduced_aliasing_early_checkpoint-vonealexnet_gaussian_noise_std4_fixed.mat', {'alexnet_early_checkpoint', 'vonealexnet_gaussian_noise_std4_fixed'}; ...
    'gvone_vs_standard_alexnet_adv_eval_l2', 'EXP9 - L2: VOneAlexNet (std=4) vs. Standard', 'EXP9_ANALYSIS/l2_alexnet_early_checkpoint-alexnet_reduced_aliasing_early_checkpoint-vonealexnet_gaussian_noise_std4_fixed.mat', {'alexnet_early_checkpoint', 'vonealexnet_gaussian_noise_std4_fixed'}; ...
    'gvone_vs_standard_alexnet_adv_eval_linf', 'EXP9 - Linf: VOneAlexNet (std=4) vs. Standard', 'EXP9_ANALYSIS/linf_alexnet_early_checkpoint-alexnet_reduced_aliasing_early_checkpoint-vonealexnet_gaussian_noise_std4_fixed.mat', {'alexnet_early_checkpoint', 'vonealexnet_gaussian_noise_std4_fixed'}; ...
    'standard_vs_lowpass_alexnet_adv_eval_l1', 'EXP9 - L1: Standard vs. LowPassAlexNet', 'EXP9_ANALYSIS/l1_alexnet_early_checkpoint-alexnet_reduced_aliasing_early_checkpoint-vonealexnet_gaussian_noise_std4_fixed.mat', {'alexnet_reduced_aliasing_early_checkpoint', 'alexnet_early_checkpoint'}; ...
    'standard_vs_lowpass_alexnet_adv_eval_l2', 'EXP9 - L2: Standard vs. LowPassAlexNet', 'EXP9_ANALYSIS/l2_alexnet_early_checkpoint-alexnet_reduced_aliasing_early_checkpoint-vonealexnet_gaussian_noise_std4_fixed.mat', {'alexnet_reduced_aliasing_early_checkpoint', 'alexnet_early_checkpoint'}; ...
    'standard_vs_lowpass_alexnet_adv_eval_linf', 'EXP9 - Linf: Standard vs. LowPassAlexNet', 'EXP9_ANALYSIS/linf_alexnet_early_checkpoint-alexnet_reduced_aliasing_early_checkpoint-vonealexnet_gaussian_noise_std4_fixed.mat', {'alexnet_reduced_aliasing_early_checkpoint', 'alexnet_early_checkpoint'}; ...
}
save_all_experiment_anovas(experiment_name, experiment_analysis_path, analysis_short_name_full_names_files_and_models, NUM_BOOTSTRAPS)
end
end

%% Helper functions for running all of the models through and saving the results

function [] = save_all_experiment_anovas(experiment_name, experiment_analysis_path, analysis_short_name_full_names_files_and_models, NUM_BOOTSTRAPS)
save_matlab_output_path = [experiment_analysis_path, '/', date, '-', experiment_analysis_path, '-', 'SMALLER_RANGE_ADV_EVAL_ANOVA_NETWORK_RESULTS.txt']

results = evalc('run_experiment(experiment_name, analysis_short_name_full_names_files_and_models, NUM_BOOTSTRAPS)')
remove_bold_results = regexprep(results, {'<strong>', '</strong>'}, {'', ''});

fid = fopen(save_matlab_output_path,'w');
fprintf(fid,'%s',remove_bold_results);
fclose(fid);
end

function [] = run_experiment(experiment_name, analysis_short_name_full_names_files_and_models, NUM_BOOTSTRAPS)
    disp(['######### | ' experiment_name ' | #########'])
    disp(['NUM_PERMUTATIONS = ' num2str(NUM_BOOTSTRAPS)])
    for analysis_idx=1:size(analysis_short_name_full_names_files_and_models,1)
        experiment_short_name = analysis_short_name_full_names_files_and_models{analysis_idx, 1}
        experiment_name = analysis_short_name_full_names_files_and_models{analysis_idx, 2}
        disp([newline ' *** | ' experiment_name ' | ***'])
        data_matrix_path = analysis_short_name_full_names_files_and_models{analysis_idx, 3};
        models =  analysis_short_name_full_names_files_and_models{analysis_idx, 4}

        [F_observer_main_effect, ...
         p_value_main_effect, ...
         F_stage_observer_interaction, ...
         p_value_interaction] = run_network_adv_eval_anova_from_data_matrix_path_smaller_range(models, ...
                                                                               experiment_short_name, ...
                                                                               data_matrix_path, ...
                                                                               NUM_BOOTSTRAPS);

    end
end

