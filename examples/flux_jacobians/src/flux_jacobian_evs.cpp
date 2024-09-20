#include <iostream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include <Eigen/Dense>

#define NROWS 8
#define NCOLS 8
#define STORAGE_PATTERN Eigen::StorageOptions::RowMajor
#define IDX3D(i, j, k, Nx, Ny, Nz) (k) * (Nx) * (Ny) + (i) * (Ny) + j

void computeA(Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &A, const float fluidvars[8]);
void computeB(Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &B, const float fluidvars[8]);
void computeC(Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &C, const float fluidvars[8]);

int main(int argc, char* argv[]){
    std::string shm_name = argv[1]; // Parent process is CUDA initialization program 
    int Nx = atoi(argv[2]);
    int Ny = atoi(argv[3]);
    int Nz = atoi(argv[4]);

    int cube_size = Nx * Ny * Nz;
    int data_size = 8 * cube_size; // 8 fluid variables, Nx*Ny*Nz values per fluidvar

    int shm_fd = shm_open(shm_name.data(), O_RDWR, 0666);
    if (shm_fd == -1){
        std::cerr << "Inside phdf5_writeall" << std::endl;
        std::cerr << "Failed to open shared memory" << std::endl;
        return EXIT_FAILURE;
    }

    float* shm_h_fluidvar = (float*)mmap(0, data_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (shm_h_fluidvar == MAP_FAILED){
        std::cerr << "Inside phdf5_writeall" << std::endl;
        std::cerr << "Failed to connect pointer to shared memory" << std::endl;
        return EXIT_FAILURE;
    }
    /* Define and instantiate Flux Jacobian Matrices */
    Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> A, B, C; // I just really like row-major order

    float fluid_point[8] = {0.0};

    for (int ifv = 0; ifv < 8; ifv++){
        fluid_point[ifv] = shm_h_fluidvar[ifv * cube_size];
    }

    computeA(A, fluid_point);
    computeB(B, fluid_point);
    computeC(C, fluid_point);

    /* Compute eigenvalues with Eigen library */
    
    // Free EVERYTHING
    munmap(shm_h_fluidvar, data_size);
    close(shm_fd);
    return 0;
}

// SPECIFY THE VALUES OF THE FLUX JACOBIANS
// Abandon all hope of non-hardcoded code, ye who enter here
// Spaghetti code is only spaghetti if there was an equivalent way of doing the same work in significantly less code
void computeA(Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &A, const float fluid_point[8]){
    float gamma = 5.0 / 3.0;

    float rho = fluid_point[0];

    float u = fluid_point[1] / rho;
    float v = fluid_point[2] / rho;
    float w = fluid_point[3] / rho;

    float Bx = fluid_point[4];
    float By = fluid_point[5];
    float Bz = fluid_point[6];

    float e = fluid_point[7];

    float usq = pow(u, 2) + pow(v, 2) + pow(w, 2);
    float Bsq = pow(Bx, 2) + pow(By, 2) + pow(Bz, 2);
    float Bdotu = Bx * u + By * v + Bz * w;

    float p = (gamma - 1.0) * (e - 0.5 * rho * usq - 0.5 * Bsq);

    // First and Fifth Row of X-Flux Jacobian 
    for (int ic = 0; ic < NCOLS; ic++){
        A(0, ic) = 0.0;
        A(4, ic) = 0.0;
    }
    A(0, 1) = 1.0;
    
    // Second Row of X-Flux Jacobian
    A(1, 0) = 0.5 * (gamma - 1.0) * usq - pow(u, 2);
    A(1, 1) = u * (3.0 - gamma); 
    A(1, 2) = v * (1.0 - gamma);
    A(1, 3) = w * (1.0 - gamma);
    A(1, 4) = -gamma * Bx;
    A(1, 5) = (2.0 - gamma) * By;
    A(1, 6) = (2.0 - gamma) * Bz;
    A(1, 7) = gamma - 1.0; 

    // Third Row of X-Flux Jacobian
    A(2, 0) = - u * v;
    A(2, 1) = v;
    A(2, 2) = u;
    A(2, 3) = 0.0;
    A(2, 4) = -By;
    A(2, 5) = -Bx;
    A(2, 6) = 0.0;
    A(2, 7) = 0.0;

    // Fourth Row of X-Flux Jacobian
    A(3, 0) = -u * w;
    A(3, 1) = w;
    A(3, 2) = 0.0;
    A(3, 3) = u;
    A(3, 4) = -Bz;
    A(3, 5) = 0.0;
    A(3, 6) = -Bx;
    A(3, 7) = 0.0;
    
    // Sixth Row of X-Flux Jacobian
    A(5, 0) = (1.0 / rho) * (v * Bx - u * By);
    A(5, 1) = By / rho;
    A(5, 2) = -Bx / rho;
    A(5, 3) = 0.0;
    A(5, 4) = -v; 
    A(5, 5) = u;
    A(5, 6) = 0.0;
    A(5, 7) = 0.0;

    // Seventh Row of X-Flux Jacobian
    A(6, 0) = (1.0 / rho) * (w * Bx - u * Bz);
    A(6, 1) = Bz / rho;
    A(6, 2) = 0.0;
    A(6, 3) = -Bx / rho;
    A(6, 4) = -w;
    A(6, 5) = 0.0;
    A(6, 6) = u;
    A(6, 7) = 0.0;

    // Eighth Row of X-Flux Jacobian
    float A_81 = u * ((gamma - 1.0) * usq - (1.0 / rho) * (gamma * e + (2 - gamma) * 0.5 * Bsq)) + Bx * (Bdotu / rho);
    float A_82 = (1.0 / rho) * (gamma * e + (2.0 - gamma) * 0.5 * Bsq) + (1.0 - gamma) * (pow(u, 2) + 0.5 * usq) - pow(Bx, 2) / rho;
    A(7, 0) = A_81;
    A(7, 1) = A_82;
    A(7, 2) = (1.0 - gamma) * u * v - (By * Bx) / rho;
    A(7, 3) = u * w * (1.0 - gamma) - (Bx * Bz) / rho;
    A(7, 4) = (1.0 - gamma) * u * Bx - Bdotu;
    A(7, 5) = (2.0 - gamma) * u * By - v * Bx;
    A(7, 6) = (2.0 - gamma) * u * Bz - w * Bx;
    A(7, 7) = u * gamma;
    
    return;
}

void computeB(Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &B, const float fluid_point[8]){
    float gamma = 5.0 / 3.0;

    float rho = fluid_point[0];

    float u = fluid_point[1] / rho;
    float v = fluid_point[2] / rho;
    float w = fluid_point[3] / rho;

    float Bx = fluid_point[4];
    float By = fluid_point[5];
    float Bz = fluid_point[6];

    float e = fluid_point[7];

    float usq = pow(u, 2) + pow(v, 2) + pow(w, 2);
    float Bsq = pow(Bx, 2) + pow(By, 2) + pow(Bz, 2);
    float Bdotu = Bx * u + By * v + Bz * w;

    float p = (gamma - 1.0) * (e - 0.5 * rho * usq - 0.5 * Bsq);

    // First Row of Y-Flux Jacobian
    
    // Second Row of Y-Flux Jacobian

    // Third Row of Y-Flux Jacobian
    
    // Fourth Row of Y-Flux Jacobian
    
    // Fifth Row of Y-Flux Jacobian
    
    // Sixth Row of Y-Flux Jacobian
    
    // Seventh Row of Y-Flux Jacobian
    
    // Eighth Row of Y-Flux Jacobian
    return;
}

void computeC(Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &C, const float fluid_point[8]){
    float gamma = 5.0 / 3.0;

    float rho = fluid_point[0];

    float u = fluid_point[1] / rho;
    float v = fluid_point[2] / rho;
    float w = fluid_point[3] / rho;

    float Bx = fluid_point[4];
    float By = fluid_point[5];
    float Bz = fluid_point[6];

    float e = fluid_point[7];

    float usq = pow(u, 2) + pow(v, 2) + pow(w, 2);
    float Bsq = pow(Bx, 2) + pow(By, 2) + pow(Bz, 2);
    float Bdotu = Bx * u + By * v + Bz * w;

    float p = (gamma - 1.0) * (e - 0.5 * rho * usq - 0.5 * Bsq);

    // First Row of Z-Flux Jacobian
    
    // Second Row of Z-Flux Jacobian

    // Third Row of Z-Flux Jacobian
    
    // Fourth Row of Z-Flux Jacobian
    
    // Fifth Row of Z-Flux Jacobian
    
    // Sixth Row of Z-Flux Jacobian
    
    // Seventh Row of Z-Flux Jacobian
    
    // Eighth Row of Z-Flux Jacobian
    return;
}