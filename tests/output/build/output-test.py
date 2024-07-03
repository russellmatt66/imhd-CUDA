import cudf

# Define problem sizes
Nxs = [2**ix for ix in range(6,10)] 
Nys = [2**iy for iy in range(6,10)]
Nzs = [2**iz for iz in range(6,10)]

Nxs[len(Nxs)-1] = 304
Nys[len(Nys)-1] = 304
Nzs[len(Nzs)-1] = 592

print(f"Nxs: {Nxs}\n Nys: {Nys}\n Nzs: {Nzs}")

# Run executable which writes data to file using custom function, and RAPIDs bindings

# Load the data from both approaches, and compare