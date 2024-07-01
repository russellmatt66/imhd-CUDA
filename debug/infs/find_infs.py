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

aln_threshold = 1e2
aln_rows = gdf[(gdf['val'] > aln_threshold) & (gdf['val'] != np.inf) & (gdf['val'] != -np.inf)]
print("Field values larger than the threshold")
print(type(aln_rows))
print(aln_rows.head())

num_alns = aln_rows.shape[0]
print(num_alns)

ninfaln_rows = cudf.concat([inf_rows, ninf_rows, aln_rows])
ninfaln_rows[['i', 'j', 'k']].to_csv('./data/ninfalns_' + var_name + nt + '.csv', index=False)

indices = ['i', 'j', 'k']
unique_values_ninfalns = {index: ninfaln_rows[index].unique() for index in indices}
for index, unique_vals in unique_values_ninfalns.items():
    print(f"Unique values in ninfaln_rows for {index}:\n {unique_vals}")

unique_values_ninfs = {index: ninf_rows[index].unique() for index in indices}
for index, unique_vals in unique_values_ninfs.items():
    print(f"Unique values in ninf_rows for {index}:\n {unique_vals}")

unique_values_infs = {index: inf_rows[index].unique() for index in indices}
for index, unique_vals in unique_values_infs.items():
    print(f"Unique values in inf_rows for {index}:\n {unique_vals}")

unique_values_alns = {index: aln_rows[index].unique() for index in indices}
for index, unique_vals in unique_values_alns.items():
    print(f"Unique values in aln_rows for {index}:\n {unique_vals}")