import pandas as pd
import json
import os
import sys
from tqdm import tqdm


script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(script_dir, '..')
sys.path.append(project_dir)

with open(f'{project_dir}/data/format/ifeval_single_instr_format.jsonl') as f:
    data = f.readlines()
    data = [json.loads(d) for d in data]
    input_data_df = pd.DataFrame(data)
    all_instructions = list(input_data_df['instruction_id_list_for_eval'].apply(lambda x: x[0]).unique())
    filters = ['detectable_format', 'language', 'change_case', 'punctuation', 'startend']
    all_instructions = list(filter(lambda x: any([f in x for f in filters]), all_instructions))

for instr in tqdm(all_instructions):
    print(instr)