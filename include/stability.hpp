#ifndef STABILITY_HPP
#define STABILITY_HPP

#include <Eigen/Dense>

#define NROWS 8
#define NCOLS 8
#define STORAGE_PATTERN Eigen::StorageOptions::RowMajor
#define IDX3D(i, j, k, Nx, Ny, Nz) (k) * (Nx) * (Ny) + (i) * (Ny) + j

/* Stability criterion */
float computeStabilityCriterionLHS(Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &A, Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &B,
    Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &C, const float fluid_point[8], const float dt, const float mesh_spacing[3]);
Eigen::Vector3f getLargestEVs(Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &A, Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &B,
    Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &C, const float fluid_point[8]);
void computeA(Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &A, const float fluid_point[8]);
void computeB(Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &B, const float fluid_point[8]);
void computeC(Eigen::Matrix<float, NROWS, NCOLS, STORAGE_PATTERN> &C, const float fluid_point[8]);

#endif