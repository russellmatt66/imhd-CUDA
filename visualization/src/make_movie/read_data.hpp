#include <string>

#include "hdf5.h" 

void getFluidvarData(float* fluidvar, const std::string file_name, const std::string dset_name);

// Rather not compute these via side-effect
hsize_t getNx(const std::string gridfile_name); 
hsize_t getNy(const std::string gridfile_name);
hsize_t getNz(const std::string gridfile_name);
