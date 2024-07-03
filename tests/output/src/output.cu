#include <string>

#include "../../../include/initialize_od.cuh"
#include "../../../include/utils.hpp"

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char* argv[]){
    int Nx = atoi(argv[1]);
    int Ny = atoi(argv[2]);
    int Nz = atoi(argv[3]);

    float x_min = -3.14159;
    float x_max = 3.14159;
    float y_min = -3.14159;
    float y_max = 3.14159;
    float z_min = -3.14159;
    float z_max = 3.14159;

    float dx = (x_max - x_min) / (Nx - 1);
	float dy = (y_max - y_min) / (Ny - 1);
	float dz = (z_max - z_min) / (Nz - 1);

    float J0 = 1.0;

    int deviceId;
	int numberOfSMs;

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    float *fluidvar;
	float *grid_x, *grid_y, *grid_z;

	int cube_size = Nx * Ny * Nz;
	int fluid_data_size = sizeof(float) * Nx * Ny * Nz;

	checkCuda(cudaMalloc(&fluidvar, 8 * fluid_data_size));
	checkCuda(cudaMalloc(&grid_x, sizeof(float) * Nx));
	checkCuda(cudaMalloc(&grid_y, sizeof(float) * Ny));
	checkCuda(cudaMalloc(&grid_z, sizeof(float) * Nz));

    int SM_mult_x = 1;
    int SM_mult_y = 1;
    int SM_mult_z = 1;

	dim3 grid_dimensions(SM_mult_x * numberOfSMs, SM_mult_y * numberOfSMs, SM_mult_z * numberOfSMs);
	dim3 block_dims_grid(32, 16, 2); // 1024 threads per block
	dim3 block_dims_init(8, 4, 4); // 256 < 923 threads per block

	InitializeGrid<<<grid_dimensions, block_dims_grid>>>(x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz,
															grid_x, grid_y, grid_z, Nx, Ny, Nz);
	checkCuda(cudaDeviceSynchronize());

	InitialConditions<<<grid_dimensions, block_dims_init>>>(fluidvar, J0, grid_x, grid_y, grid_z, Nx, Ny, Nz); // Screw-pinch
	checkCuda(cudaDeviceSynchronize());

    // Prepare host data for writing out
	std::vector<std::string> fluid_data_files (8); // 8 is the number of threads I'm going with
    std::string base_file = "../data/rho/";
    for (size_t i = 0; i < fluid_data_files.size(); i++){
        fluid_data_files[i] = base_file + std::to_string(i) + ".csv";
    }   

	float *h_rho, *h_rhovx, *h_rhovy, *h_rhovz, *h_Bx, *h_By, *h_Bz, *h_e;

	h_rho = (float*)malloc(fluid_data_size);
	h_rhovx = (float*)malloc(fluid_data_size);
	h_rhovy = (float*)malloc(fluid_data_size);
	h_rhovz = (float*)malloc(fluid_data_size);
	h_Bx = (float*)malloc(fluid_data_size);
	h_By = (float*)malloc(fluid_data_size);
	h_Bz = (float*)malloc(fluid_data_size);
	h_e = (float*)malloc(fluid_data_size);

    int* to_write_or_not;
	to_write_or_not = (int*)malloc(8 * sizeof(int));

	for (int i = 0; i < 8; i++){ /* COULD USE A CHAR FOR THIS */
		to_write_or_not[i] = atoi(argv[4 + i]);
	}

    for (size_t ih = 0; ih < 8; ih++){
		if (!to_write_or_not[ih]){ // No need for the host memory if it's not being written out
			switch (ih)
			{
			case 0:
				free(h_rho);
				break;
			case 1:
				free(h_rhovx);
				break;
			case 2:
				free(h_rhovy);
				break;			
			case 3:
				free(h_rhovz);
				break;			
			case 4:
				free(h_Bx);
				break;			
			case 5:
				free(h_By);
				break;			
			case 6:
				free(h_Bz);
				break;			
			case 7:
				free(h_e);
				break;			
			default:
				break;
			}
		}
	}

    for (size_t iv = 0; iv < 8; iv++){ 
        if (to_write_or_not[iv]){  
            switch (iv)
            {
            case 0:
                cudaMemcpy(h_rho, fluidvar, fluid_data_size, cudaMemcpyDeviceToHost);
                break;
            case 1:
                // cudaMemcpy(h_rhovx, fluidvar + fluid_data_size, fluid_data_size, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_rhovx, fluidvar + cube_size, fluid_data_size, cudaMemcpyDeviceToHost);
                break;
            case 2:
                // cudaMemcpy(h_rhovy, fluidvar + 2 * fluid_data_size, fluid_data_size, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_rhovy, fluidvar + 2 * cube_size, fluid_data_size, cudaMemcpyDeviceToHost);
                break;
            case 3:
                // cudaMemcpy(h_rhovz, fluidvar + 3 * fluid_data_size, fluid_data_size, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_rhovz, fluidvar + 3 * cube_size, fluid_data_size, cudaMemcpyDeviceToHost);
                break;			
            case 4:
                // cudaMemcpy(h_Bx, fluidvar + 4 * fluid_data_size, fluid_data_size, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_Bx, fluidvar + 4 * cube_size, fluid_data_size, cudaMemcpyDeviceToHost);
                break;				
            case 5:
                // cudaMemcpy(h_By, fluidvar + 5 * fluid_data_size, fluid_data_size, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_By, fluidvar + 5 * cube_size, fluid_data_size, cudaMemcpyDeviceToHost);
                break;				
            case 6:
                // cudaMemcpy(h_Bz, fluidvar + 6 * fluid_data_size, fluid_data_size, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_Bz, fluidvar + 6 * cube_size, fluid_data_size, cudaMemcpyDeviceToHost);
                break;				
            case 7:
                // cudaMemcpy(h_e, fluidvar + 7 * fluid_data_size, fluid_data_size, cudaMemcpyDeviceToHost);
                cudaMemcpy(h_e, fluidvar + 7 * cube_size, fluid_data_size, cudaMemcpyDeviceToHost);
                break;				
            default:
                break;
            }
        }
    }
    checkCuda(cudaDeviceSynchronize());

    for (size_t iv = 0; iv < 8; iv++){ 
        if (to_write_or_not[iv]){ 
            base_file = getNewBaseDataLoc(iv);
            for (size_t i = 0; i < fluid_data_files.size(); i++){
                fluid_data_files[i] = base_file + "_" + std::to_string(i) + ".csv";
            }  
            switch (iv)
            {
                case 0:
                    writeFluidVars(fluid_data_files, h_rho, Nx, Ny, Nz);					
                    break;
                case 1:
                    writeFluidVars(fluid_data_files, h_rhovx, Nx, Ny, Nz);					
                    break;
                case 2:
                    writeFluidVars(fluid_data_files, h_rhovy, Nx, Ny, Nz);					
                    break;
                case 3:
                    writeFluidVars(fluid_data_files, h_rhovz, Nx, Ny, Nz);					
                    break;			
                case 4:
                    writeFluidVars(fluid_data_files, h_Bx, Nx, Ny, Nz);					
                    break;				
                case 5:
                    writeFluidVars(fluid_data_files, h_By, Nx, Ny, Nz);					
                    break;				
                case 6:
                    writeFluidVars(fluid_data_files, h_Bz, Nx, Ny, Nz);					
                    break;				
                case 7:
                    writeFluidVars(fluid_data_files, h_e, Nx, Ny, Nz);					
                    break;				
                default:
                    break;
            }
        }
    }
    checkCuda(cudaDeviceSynchronize());
    return 0;
}