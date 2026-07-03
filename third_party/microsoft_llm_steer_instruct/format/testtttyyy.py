import pandas as pd
df = pd.read_hdf('pre_computed_ivs_layer_20.h5', key='df')
print('Columns in file:', df.columns.tolist())