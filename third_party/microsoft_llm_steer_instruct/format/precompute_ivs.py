# %%
import os
import sys
import json
import pandas as pd
import torch
from tqdm import tqdm
from omegaconf import DictConfig
import hydra

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.append(project_dir)

config_path = os.path.join(project_dir, 'config/format')

from utils.generation_utils import compute_task_matrix

# Pulled from Stolfo et al. Table 8 (format) + Table 9 (language)
# Missing keys = dash in the paper (steering found no benefit) -> treat as skip (-1).
# kn, ko, ne are excluded entirely: not present in Table 9 at all, no ground truth exists.

GEMMA_2B_W_INSTR = {
    'change_case:capital_word_frequency': 23,
    'change_case:english_capital': 11,
    'change_case:english_lowercase': 7,
    'detectable_format:json_format': 13,
    'detectable_format:multiple_sections': 9,
    'punctuation:no_comma': 5,
    'startend:end_checker': 5,
    'startend:quotation': 23,
    'language:response_language_ar': 15,
    'language:response_language_hi': 7,
    'language:response_language_mr': 15,
    # Note: For languages like De, Ru, Vi, the table has dashes (-) 
    # indicating steering was unnecessary/unhelpful.
}

# Use these if args.include_instructions = False
GEMMA_2B_WO_INSTR = {
    'change_case:capital_word_frequency': 21,
    'change_case:english_capital': 11,
    'change_case:english_lowercase': 17,
    'detectable_format:number_bullet_lists': 5,
    'detectable_format:number_highlighted_sections': 5,
    'punctuation:no_comma': 11,
    'startend:quotation': 11,
    'language:response_language_bg': 19,
    'language:response_language_de': 15,
    'language:response_language_fi': 15,
    'language:response_language_hi': 15,
    'language:response_language_it': 15,
    'language:response_language_mr': 17,
    'language:response_language_pt': 15,
    'language:response_language_ru': 15,
    'language:response_language_vi': 15,
}

PHI3_WO_INSTR = {
    'punctuation:no_comma': 10,
    'change_case:english_lowercase': 18,
    'detectable_format:number_bullet_lists': 20,
    'startend:quotation': 26,
    'language:response_language_mr': 30,
    'detectable_format:number_highlighted_sections': 26,
    'language:response_language_fa': 20,
    'change_case:english_capital': 28,
    'change_case:capital_word_frequency': 18,
    'language:response_language_sw': 20,
    'language:response_language_ru': 16,
    'language:response_language_hi': 22,
    'language:response_language_ar': 18,
    'language:response_language_ur': 28,
    'language:response_language_de': 16,
    # dashes (skip): detectable_format:json_format, startend:end_checker,
    # detectable_format:multiple_sections, language:response_language_pa,
    # detectable_format:title, detectable_format:constrained_response,
    # language:response_language_gu, language:response_language_te
}

PHI3_W_INSTR = {
    'punctuation:no_comma': 12,
    'change_case:english_lowercase': 30,
    'detectable_format:number_bullet_lists': 16,
    'detectable_format:json_format': 6,
    'startend:end_checker': 18,
    'detectable_format:multiple_sections': 8,
    'startend:quotation': 24,
    'language:response_language_mr': 14,
    'detectable_format:number_highlighted_sections': 18,
    'change_case:english_capital': 22,
    'change_case:capital_word_frequency': 20,
    'language:response_language_hi': 14,
    'language:response_language_gu': 14,
    'language:response_language_ar': 16,
    # dashes (skip): language:response_language_pa, detectable_format:title,
    # language:response_language_fa, detectable_format:constrained_response,
    # language:response_language_sw, language:response_language_ru,
    # language:response_language_te, language:response_language_ur,
    # language:response_language_de
}

@hydra.main(config_path=config_path, config_name='precompute_steering_vectors')
def precompute_vectors(args: DictConfig):
    with open(f'{project_dir}/data/format/ifeval_single_instr_format.jsonl') as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]
    input_data_df = pd.DataFrame(data)
    all_instructions = list(input_data_df['instruction_id_list_for_eval'].apply(lambda x: x[0]).unique())

    # filter out instructions that are not detectable_format, language, change_case, punctuation, or startend
    filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
    all_instructions = list(filter(lambda x: any([f in x for f in filters]), all_instructions))

    w_perplexity = '_with_perplexity' if args.use_perplexity else ''
    cross_model = '_cross_model' if args.cross_model_steering else ''
    instr_included = 'instr' if args.include_instructions else 'no_instr'

    # --------------------------------------------------------
    # This code is commented out since this is relying on the layer sweep to flush out. The entire point of this loop
    # is to identify the optimal layer FOR EACH INSTRUCTION in the variable optimal_layers. This can be circumvented if an explicit
    # layer to evaluate is passed in. Layer sweep only identifies the layers the computation precompute_ivys can do anyways w/ layer specification.
    # --------------------------------------------------------
    folder = f'{script_dir}/layer_search_out'
    file = f'{folder}/{args.model_name}/n_examples{args.n_examples}_seed{args.seed}_{args.steering}{cross_model}{w_perplexity}/out_{instr_included}.jsonl'
    with open(file, 'r') as f:
        results = [json.loads(line) for line in f]

    validation_df = pd.DataFrame(results)
    optimal_layers = { instr: -1 for instr in all_instructions }

    for instr in all_instructions:
        if instr not in validation_df.single_instruction_id.unique():
            optimal_layers[instr] = -1
            continue

        instr_df = validation_df[validation_df.single_instruction_id == instr]
        
        if args.use_perplexity:
            # add boolean column that is true when perplexity is low
            instr_df['low_perplexity'] = instr_df.perplexity < args.preplexity_threshold

            df_group_by_layer = instr_df[['layer', 'follow_all_instructions', 'low_perplexity']].groupby('layer').mean()

            if args.model_name == 'gemma-2-9b' or args.model_name == 'gemma-2-2b':
                baseline_low_perplexity = df_group_by_layer.loc[-1, 'low_perplexity']
            else:
                baseline_low_perplexity = 0

            # get accuracy for layer -1
            accuracy_layer_minus_1 = df_group_by_layer.loc[-1, 'follow_all_instructions']

            df_group_by_layer.loc[df_group_by_layer.low_perplexity > baseline_low_perplexity, 'follow_all_instructions'] = 0

            # restore accuracy for layer -1
            df_group_by_layer.loc[-1, 'follow_all_instructions'] = accuracy_layer_minus_1

            df_group_by_layer.loc[df_group_by_layer.low_perplexity > baseline_low_perplexity, 'follow_all_instructions'] = 0
            max_accuracy = df_group_by_layer.follow_all_instructions.max()
            optimal_layer = df_group_by_layer[df_group_by_layer.follow_all_instructions == max_accuracy].index
            optimal_layers[instr] = optimal_layer[0]

        else:
            max_accuracy = instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean().follow_all_instructions.max()
            optimal_layer = instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean()[instr_df[['layer', 'follow_all_instructions']].groupby('layer').mean().follow_all_instructions == max_accuracy].index
            optimal_layers[instr] = optimal_layer[0]

    rows = []

    for instr in tqdm(all_instructions):
        # check if the file exists
        if args.model_name == 'gemma-2-2b' and args.cross_model_steering:
            print('Using representations from gemma-2-2b-it')
            rep_folder = f'{script_dir}/representations/gemma-2-2b-it/{args.representations_folder}'
        elif args.model_name == 'gemma-2-9b' and args.cross_model_steering:
            print('Using representations from gemma-2-9b-it')
            rep_folder = f'{script_dir}/representations/gemma-2-9b-it/{args.representations_folder}'
        else:
            rep_folder = f'{script_dir}/representations/{args.model_name}/{args.representations_folder}'

        file =f'{rep_folder}/{"".join(instr).replace(":", "_")}.h5'
        
        if not os.path.exists(file):
            print(f'File {file} does not exist')
            continue
        results_df = pd.read_hdf(file, key='df')

        row = {}
        row['instruction'] = instr

        hs_instr = results_df['last_token_rs'].to_list()
        hs_instr = torch.tensor(hs_instr, device=args.device)
        hs_no_instr = results_df['last_token_rs_no_instr'].to_list()
        hs_no_instr = torch.tensor(hs_no_instr, device=args.device)

        # check if hs has 4 dimensions
        if len(hs_instr.shape) == 3:
            hs_instr = hs_instr.unsqueeze(2)
            hs_no_instr = hs_no_instr.unsqueeze(2)

        if args.include_instructions:
            TASK_LAYER_MAP = PHI3_W_INSTR
        else:
            TASK_LAYER_MAP = PHI3_WO_INSTR

        # # Dynamic layer selection per task (refactor)
        # # if instr in TASK_LAYER_MAP:
        # #     selected_layer = TASK_LAYER_MAP[instr]
        # if args.specific_layer is not None:
        #     selected_layer = args.specific_layer
        # else:
        #     # Commented out since layer sweep is not being used right now
        #     selected_layer = optimal_layers[instr]
        #     # selected_layer = -1 # Default fallback
        
        # mult_rs's layer comes from the real sweep (optimal_layers) -- unchanged
        if args.specific_layer is not None:
            selected_layer = args.specific_layer
        else:
            selected_layer = optimal_layers[instr]

        # adjust_rs / add_vector's layer comes from the paper's Table 8/9 value --
        # completely independent of whatever mult_rs's sweep found
        adjust_layer = TASK_LAYER_MAP.get(instr, -1)

        repr_diffs = hs_instr - hs_no_instr
        mean_repr_diffs = repr_diffs.mean(dim=0)
        last_token_mean_diff = mean_repr_diffs[:, -1, :]

        # ---- mult_rs: task matrix at ITS OWN layer ----
        if selected_layer == -1:
            task_matrix = torch.eye(hs_instr.shape[-1])
        else:
            X = hs_no_instr[:, selected_layer, -1, :]
            X_plus = hs_instr[:, selected_layer, -1, :]
            task_matrix = compute_task_matrix(X, X_plus)

        # ---- adjust_rs / add_vector: direction + avg_proj at THEIR OWN layer ----
        if adjust_layer == -1:
            instr_dir = torch.zeros(hs_instr.shape[-1])
            avg_proj = torch.tensor(0.0)
            avg_proj_no_instr = torch.tensor(0.0)
        else:
            instr_dir = last_token_mean_diff[adjust_layer] / last_token_mean_diff[adjust_layer].norm()
            X_adj = hs_no_instr[:, adjust_layer, -1, :]
            X_plus_adj = hs_instr[:, adjust_layer, -1, :]
            avg_proj = (X_plus_adj.to(args.device) @ instr_dir.to(args.device)).mean()
            avg_proj_no_instr = (X_adj.to(args.device) @ instr_dir.to(args.device)).mean()

        print(f"Number of training instances for: {instr}")
        print(f"Number of training instances (N): {hs_instr.shape[0]}")
        print(f"Activation feature dimensionality (D): {hs_instr.shape[-1]}")

        row['selected_layer_mult_rs'] = selected_layer
        row['selected_layer_adjust_rs'] = adjust_layer
        row['selected_layer_add_vector'] = adjust_layer
        row['task_matrix'] = task_matrix.cpu().numpy() if torch.is_tensor(task_matrix) else task_matrix
        row['instr_dir'] = instr_dir.cpu().numpy() if torch.is_tensor(instr_dir) else instr_dir
        row['avg_proj'] = avg_proj
        row['avg_proj_no_instr'] = avg_proj_no_instr

        rows.append(row)

    df = pd.DataFrame(rows)

    # store the df in folder
    folder = f'{script_dir}/representations/{args.model_name}/{args.representations_folder}'
    if args.specific_layer is not None:
        df.to_hdf(f'{folder}/pre_computed_ivs_layer_{args.specific_layer}.h5', key='df', mode='w')
    else:
        df.to_hdf(f'{folder}/pre_computed_ivs_best_layer_validation{w_perplexity}{cross_model}_{instr_included}.h5', key='df', mode='w')
            
if __name__ == '__main__':
    precompute_vectors()
# %%
