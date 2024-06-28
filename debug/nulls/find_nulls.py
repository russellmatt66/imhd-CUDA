# Find the locations where there are null values in the output data
import sys
try:
    import cudf
except ModuleNotFoundError:
    pass
import pandas as pd

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
except NameError:
    gdf = pd.read_csv(out_file)

null_rows = gdf[gdf['val'].isnull()]
print(type(null_rows))
print(null_rows.head())

num_nulls = null_rows.shape[0]
print(num_nulls)

null_rows[['i', 'j', 'k']].to_csv('./data/nulls_' + var_name + nt + '.csv', index=False)
# print(null_rows['i'].unique())
# indices = ['i', 'j', 'k']

# unique_values = {index: null_rows[index].unique() for index in indices}

# for index, unique_vals in unique_values.items():
#     print(f"Unique values for {index}: {unique_vals}")