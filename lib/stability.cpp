#include <thread>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#include "stability.hpp"

#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx * Ny) + (i) * Ny + j) // row-major, column-minor order

/* Stability criterion */
float computeStabilityCriterionLHS(Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &A, Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &B,
    Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &C, const float fluid_point[8], const float dt, const float mesh_spacing[3]){
        Eigen::Vector3cf largest_evs = getLargestEVs(A, B, C, fluid_point);
        float dx = mesh_spacing[0], dy = mesh_spacing[1], dz = mesh_spacing[2];

        return (dt / dx) * abs(largest_evs[0]) + (dt / dy) * abs(largest_evs[1]) + (dt / dz) * abs(largest_evs[2]);
    }

Eigen::Vector3f getLargestEVs(Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &A, Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &B,
    Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &C, const float fluid_point[8]){
        Eigen::Vector3f largest_evs (0.0, 0.0, 0.0); // Ideal MHD is a hyperbolic system - real eigenvalues
        
        computeA(A, fluid_point);
        computeB(B, fluid_point);
        computeC(C, fluid_point);

        Eigen::VectorXcf eivals_A = A.eigenvalues(), eivals_B = B.eigenvalues(), eivals_C = C.eigenvalues();

        Eigen::VectorXf evA_mag = eivals_A.array().abs(), evB_mag = eivals_B.array().abs(), evC_mag = eivals_C.array().abs();

        largest_evs[0] = evA_mag.maxCoeff();
        largest_evs[1] = evB_mag.maxCoeff();
        largest_evs[2] = evC_mag.maxCoeff();

        return largest_evs;
    }

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

    // First and Sixth Rows of Y-Flux Jacobian
    for (int ic = 0; ic < NCOLS; ic++){
        B(0, ic) = 0.0;
        B(5, ic) = 0.0;
    }
    B(0, 2) = 1.0;
    
    // Second Row of Y-Flux Jacobian
    B(1,0) = -u * v;
    B(1,1) = v;
    B(1,2) = u;
    B(1,3) = 0.0;
    B(1,4) = -By;
    B(1,5) = -Bx;
    B(1,6) = 0.0;
    B(1,7) = 0.0;

    // Third Row of Y-Flux Jacobian
    B(2,0) = 0.5 * (gamma - 1.0) * usq - pow(v,2); 
    B(2,1) = (1.0 - gamma) * u;
    B(2,2) = (3.0 - gamma) * v;
    B(2,3) = (1.0 - gamma) * w;
    B(2,4) = (2.0 - gamma) * Bx;
    B(2,5) = -gamma * By;
    B(2,6) = (2.0 - gamma) * Bz;
    B(2,7) = gamma - 1.0;

    // Fourth Row of Y-Flux Jacobian
    B(3,0) = -v * w;
    B(3,1) = 0.0;
    B(3,2) = w;
    B(3,3) = v;
    B(3,4) = 0.0;
    B(3,5) = -Bz;
    B(3,6) = -By;
    B(3,7) = 0.0;

    // Fifth Row of Y-Flux Jacobian
    B(4,0) = (1.0 / rho) * (v * Bx - u * By);
    B(4,1) = By / rho;
    B(4,2) = -Bx / rho;
    B(4,3) = 0.0;
    B(4,4) = -v; 
    B(4,5) = u;
    B(4,6) = 0.0;
    B(4,7) = 0.0;

    // Seventh Row of Y-Flux Jacobian
    B(6,0) = (1.0 / rho) * (v * Bz - w * By);
    B(6,1) = 0.0;
    B(6,2) = -Bz / rho;
    B(6,3) = By / rho;
    B(6,4) = 0.0;
    B(6,5) = w;
    B(6,6) = -v;
    B(6,7) = 0.0;

    // Eighth Row of Y-Flux Jacobian
    float B_81 = v * ((gamma - 1.0) * usq - (1.0 / rho) * (gamma * e + (2.0 - gamma) * Bsq * 0.5)) + By * Bdotu / rho;
    float B_83 = (1.0 / rho) * (gamma * e + (2.0 - gamma) * Bsq * 0.5) + (1.0 - gamma) * (pow(v,2) + 0.5 * usq) - pow(By, 2) / rho;
    B(7,0) = B_81;
    B(7,1) = (1.0 - gamma) * u * v - Bx * By / rho;
    B(7,2) = B_83;
    B(7,3) = (1.0 - gamma) * v * w - By * Bz / rho;
    B(7,4) = (2.0 - gamma) * v * Bx - u * By;
    B(7,5) = (1.0 - gamma) * v * By + By * Bdotu;
    B(7,6) = (2.0 - gamma) * v * Bz - w * By;
    B(7,7) = v * gamma;
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

    // First and Seventh Row of Z-Flux Jacobian
    for (int ic = 0; ic < NCOLS; ic++){
        C(0, ic) = 0.0;
        C(6, ic) = 0.0;
    }
    C(0, 3) = 1.0;
    
    // Second Row of Z-Flux Jacobian
    C(1,0) = -u * w;
    C(1,1) = w;
    C(1,2) = 0.0;
    C(1,3) = u;
    C(1,4) = -Bz;
    C(1,5) = 0.0;
    C(1,6) = -Bx;
    C(1,7) = 0.0;

    // Third Row of Z-Flux Jacobian
    C(2,0) = -v * w;
    C(2,1) = 0.0;
    C(2,2) = w;
    C(2,3) = v;
    C(2,4) = -Bz;
    C(2,5) = 0.0;
    C(2,6) = -Bx;
    C(2,7) = 0.0;

    // Fourth Row of Z-Flux Jacobian
    C(3,0) = 0.5 * (gamma - 1.0) * usq - pow(w,2);
    C(3,1) = (1.0 - gamma) * u;
    C(3,2) = (1.0 - gamma) * v;
    C(3,3) = (3.0 - gamma) * w;
    C(3,4) = (2.0 - gamma) * Bx;
    C(3,5) = (2.0 - gamma) * By;
    C(3,6) = -gamma * By;
    C(3,7) = gamma - 1.0; 

    // Fifth Row of Z-Flux Jacobian
    C(4,0) = (1.0 / rho) * (w * Bx - u * Bz);
    C(4,1) = Bz / rho;
    C(4,2) = 0.0;
    C(4,3) = -Bx/ rho;
    C(4,4) = -w;
    C(4,5) = 0.0;
    C(4,6) = u;
    C(4,7) = 0.0;

    // Sixth Row of Z-Flux Jacobian
    C(5,0) = (1.0 / rho) * (w * By - v * Bz);
    C(5,1) = 0.0;
    C(5,2) = Bz / rho;
    C(5,3) = -By / rho;
    C(5,4) = 0.0;
    C(5,5) = -w;
    C(5,6) = v;
    C(5,7) = 0.0;    
    
    // Eighth Row of Z-Flux Jacobian
    float C_81 = w * ((gamma - 1.0) * usq - (1.0 / rho) * (gamma * e + (2.0 - gamma) * Bsq * 0.5)) + Bz * Bdotu / rho;
    float C_84 = (1.0 / rho) * (gamma * e + (2.0 - gamma) * Bsq * 0.5) + (1.0 - gamma) * (pow(w,2) + 0.5 * usq) - pow(Bz, 2) / rho;
    C(7,0) = C_81;
    C(7,1) = (1.0 - gamma) * u * w - Bx * Bz / rho;
    C(7,2) = (1.0 - gamma) * v * w - By * Bz / rho;
    C(7,3) = C_84;
    C(7,4) = (2.0 - gamma) * w * Bx - u * Bz;
    C(7,5) = (2.0 - gamma) * w * By - v * Bz;
    C(7,6) = (1.0 - gamma) * w * Bz - Bdotu;
    C(7,7) = w * gamma; 

    return;
}