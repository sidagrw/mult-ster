import pandas as pd
# Path to your manual suitcase
path = "format/representations/google/gemma-2-2b-it/all/pre_computed_ivs_layer14.h5"
df = pd.read_hdf(path)
print("🧐 YOUR SUITCASE CONTAINS THESE NAMES:")
print(df['instruction'].unique()[:10]) # Print first 10