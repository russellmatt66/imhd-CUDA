# WIP
# Use RAPIDs to analyze solution data for infs / ALNs 

import sys
import cudf
import numpy as np

out_file = sys.argv[1]
var_name = out_file.split('data/')[1].split('/var_')[0]
print(var_name)
nt = out_file.split('var_')[1].split('.csv')[0]
print(nt)

try:
    gdf = cudf.read_csv(out_file)
except FileNotFoundError:
    print("Argument needs to provide full path to the data file")
    sys.exit()

inf_rows = gdf[gdf['val'] == np.inf]
print("Field values with value of inf")
print(type(inf_rows))
print(inf_rows.head())

num_infs = inf_rows.shape[0]
print(num_infs)

ninf_rows = gdf[gdf['val'] == -np.inf]
print("Field values with value of -inf")
print(type(ninf_rows))
print(ninf_rows.head())

num_ninfs = ninf_rows.shape[0]
print(num_ninfs)

aln_threshold = 1e10
aln_rows = gdf[gdf['val'] > aln_threshold]
print("Field values larger than the threshold")
print(type(aln_rows))
print(aln_rows.head())

num_alns = aln_rows.shape[0]
print(num_alns)

ninfaln_rows = cudf.concat([inf_rows, ninf_rows, aln_rows])

ninfaln_rows[['i', 'j', 'k']].to_csv('./data/ninfalns_' + var_name + nt + '.csv', index=False)
# print(null_rows['i'].unique())
# indices = ['i', 'j', 'k']

# unique_values = {index: null_rows[index].unique() for index in indices}

# for index, unique_vals in unique_values.items():
#     print(f"Unique values for {index}: {unique_vals}")