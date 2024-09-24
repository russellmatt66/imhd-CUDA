#include <iostream>
#include <string>

#include "hdf5.h"

#define IDX3D(i, j, k, Nx, Ny, Nz) (k) * (Nx) * (Ny) + (i) * (Ny) + j
/* 
Visualizes the initial density, rho
Future work can implement feature for visualizing any of the variables
*/
int main(int argc, char* argv[]){
    std::string file_name = argv[1];
    std::string dset_name = argv[2]; // This will be called from within a Python context, so the dataset name will be known
    // int Nx = atoi(argv[3]);
    // int Ny = atoi(argv[4]);
    // int Nz = atoi(argv[5]);

    /* Read .h5 dataset data into buffer */
    hid_t file_id, dset_id, dspc_id;

    file_id = H5Fopen(file_name.data(), H5F_ACC_RDONLY, H5P_DEFAULT);

    dset_id = H5Dopen(file_id, dset_name.data(), H5P_DEFAULT);
    dspc_id = H5Dget_space(dset_id);
    int num_points = H5Sget_simple_extent_npoints(dspc_id);

    float* fluid_var;
    fluid_var = (float*)malloc(sizeof(float) * num_points);

    H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fluid_var);

    int num_attrs = H5Aget_num_attrs(dset_id);
    // Iterate over attributes and print their names and values
    for (int i = 0; i < num_attrs; ++i) {
        // Open the attribute by index
        hid_t attr_id = H5Aopen_by_idx(dset_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, (hsize_t)i, H5P_DEFAULT, H5P_DEFAULT);

        // Get the name of the attribute
        char attr_name[256];
        H5Aget_name(attr_id, sizeof(attr_name), attr_name);
        printf("Attribute %d: %s\n", i + 1, attr_name);

        // Get the datatype and dataspace of the attribute
        hid_t attr_type = H5Aget_type(attr_id);
        hid_t attr_space = H5Aget_space(attr_id);

        // Get the attribute size and type (assuming a scalar attribute for simplicity)
        hsize_t attr_size;
        H5Sget_simple_extent_dims(attr_space, &attr_size, NULL);

        // For this example, assume attributes are strings or integers
        if (H5Tget_class(attr_type) == H5T_STRING) {
            char attr_value[256];
            H5Aread(attr_id, attr_type, attr_value);
            printf("  Value: %s\n", attr_value);
        } else if (H5Tget_class(attr_type) == H5T_INTEGER) {
            int attr_value;
            H5Aread(attr_id, attr_type, &attr_value);
            printf("  Value: %d\n", attr_value);
        }

        // Close the attribute
        H5Sclose(attr_space);
        H5Tclose(attr_type);
        H5Aclose(attr_id);
    }

    /* VTK rendering pipeline */

    // Free everything
    H5Fclose(file_id);
    H5Dclose(dset_id);
    H5Dclose(dspc_id);
    free(fluid_var);
    return 0;
}