import os
import sys
import torch
import pandas as pd
import tqdm
import json
from omegaconf import DictConfig, OmegaConf
import hydra
import functools
from transformer_lens import utils as tlutils


script_dir = os.path.dirname(os.path.abspath(__file__))

model_name = ""

folder = f'{script_dir}/representations'
file_path = f'{folder}/pre_computed_ivs_layer_20.h5'
pre_computed_ivs = pd.read_hdf(file_path, key='df')
pre_computed_ivs.to_csv('sanity.csv', index=False)

for index, row in pre_computed_ivs.iterrows():
    # Access fields using dict-style syntax
    print(f"Index: {index}, Instruction: {row['instruction']}, Selected Layer: {row['selected_layer']}")
