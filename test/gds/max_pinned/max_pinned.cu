#include <cufile.h>
#include <iostream>

#include "cufile_sample_utils.h"

int main(int argc, char* argv[]){
    CUfileDrvProps_t driver_properties;
    CUfileError_t status;

    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "Failed to open cuFileDriver successfully: " << cuFileGetErrorString(status) << std::endl;
        return 1;
    }

    status = cuFileDriverGetProperties(&driver_properties);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "Failed to get cuFile driver properties: " << cuFileGetErrorString(status) << std::endl;
        return 1;
    }

    std::cout << "Maximum pinned memory size on GPU: " << driver_properties.max_device_pinned_mem_size / (1024 * 1024) << " MB" << std::endl;

    size_t max_pinned_memory_cust = UINT64_MAX; // KB 
    status = cuFileDriverSetMaxPinnedMemSize(max_pinned_memory_cust);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "Failed to set max pinned size: " << cuFileGetErrorString(status) << std::endl;
        return 1;
    }

    status = cuFileDriverGetProperties(&driver_properties);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "Failed to get cuFile driver properties: " << cuFileGetErrorString(status) << std::endl;
        return 1;
    }

    std::cout << "Maximum pinned memory size on GPU: " << driver_properties.max_device_pinned_mem_size / (1024 * 1024) << " MB" << std::endl;

    status = cuFileDriverClose();
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "Failed to close cuFileDriver: " << cuFileGetErrorString(status) << std::endl;
        return 1;
    }
    return 0;
}
