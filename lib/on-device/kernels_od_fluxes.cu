#include "helper_functions.cuh"

// /* THIS NEEDS TO BE PLACED IN A FILE SOMEWHERE SO THERE'S ONLY A SINGLE DEFINITION */
// // row-major, column-minor order
#define IDX3D(i, j, k, Nx, Ny, Nz) ((k) * (Nx * Ny) + (i) * Ny + j) // parentheses are necessary to avoid calculating `i - 1 * Ny` or `k - 1 * (Nx * Ny)`

// X-Fluxes
__device__ float XFluxRho(const float rhovx){
    return rhovx;
}

__device__ float XFluxRhoVX(const float rho, const float rhovx, const float Bx, const float p, const float Bsq){
    return pow(rhovx, 2) / rho - pow(Bx, 2) + p + 0.5 * Bsq ;
}

__device__ float XFluxRhoVY(const float rho, const float rhovx, const float rhovy, const float Bx, const float By){
    return (rhovx * rhovy) / rho - Bx * By;
}

__device__ float XFluxRhoVZ(const float rho, const float rhovx, const float rhovz, const float Bx, const float Bz){
    return (rhovx * rhovz) / rho - Bx * Bz;
}

__device__ float XFluxBX(){
    return 0.0;
}

__device__ float XFluxBY(const float rho, const float rhovx, const float rhovy, const float Bx, const float By){
    return (rhovy / rho) * Bx - (rhovx / rho) * By;
}

__device__ float XFluxBZ(const float rho, const float rhovx, const float rhovz, const float Bx, const float Bz){
    return (rhovz / rho) * Bx - (rhovx / rho) * Bz;
}

__device__ float XFluxE(const float rho, const float rhovx, const float Bx, const float e, const float p, const float Bsq, const float Bdotu){
    return (e + p + 0.5 * Bsq) * (rhovx / rho) - Bdotu * Bx;
}

// Y-Fluxes
__device__ float YFluxRho(const float rhovy){
    return rhovy;
}

__device__ float YFluxRhoVX(const float rho, const float rhovx, const float rhovy, const float Bx, const float By){
    return (rhovx * rhovy) / rho - Bx * By;
}

__device__ float YFluxRhoVY(const float rho, const float rhovy, const float By, const float p, const float Bsq){
    return pow(rhovy, 2) / rho - pow(By, 2) + p + 0.5 * Bsq;
}

__device__ float YFluxRhoVZ(const float rho, const float rhovy, const float rhovz, const float By, const float Bz){
    return (rhovy * rhovz) / rho - By * Bz;
}

__device__ float YFluxBX(const float rho, const float rhovx, const float rhovy, const float Bx, const float By){
    return (rhovx / rho) * By - (rhovy / rho) * Bx;
}

__device__ float YFluxBY(){
    return 0.0;
}

__device__ float YFluxBZ(const float rho, const float rhovy, const float rhovz, const float By, const float Bz){
    return (rhovz / rho) * By - (rhovy / rho) * Bz;
}

__device__ float YFluxE(const float rho, const float rhovy, const float By, const float e, const float p, const float Bsq, const float Bdotu){
    return (e + p + 0.5 * Bsq) * (rhovy / rho) - Bdotu * By;
}

// Z-Fluxes
__device__ float ZFluxRho(const float rhovz){
    return rhovz;
}

__device__ float ZFluxRhoVX(const float rho, const float rhovx, const float rhovz, const float Bx, const float Bz){
    return (rhovx * rhovz) / rho - Bx * Bz;
}

__device__ float ZFluxRhoVY(const float rho, const float rhovy, const float rhovz, const float By, const float Bz){
    return (rhovy * rhovz) / rho - By * Bz;
}

__device__ float ZFluxRhoVZ(const float rho, const float rhovz, const float Bz, const float p, const float Bsq){
    return pow(rhovz, 2) / rho - pow(Bz, 2) + p + 0.5 * Bsq;
}

__device__ float ZFluxBX(const float rho, const float rhovx, const float rhovz, const float Bx, const float Bz){
    return (rhovx / rho) * Bz - (rhovz / rho) * Bx;
}

__device__ float ZFluxBY(const float rho, const float rhovy, const float rhovz, const float By, const float Bz){
    return (rhovy / rho) * Bz - (rhovz / rho) * By;
}

__device__ float ZFluxBZ(){
    return 0.0;
}

__device__ float ZFluxE(const float rho, const float rhovz, const float Bz, const float e, const float p, const float Bsq, const float Bdotu){
    return (e + p + 0.5 * Bsq) * (rhovz / rho) - Bdotu * Bz;
}

/* 
I THINK THIS IS WHAT'S CAUSING THE CACHE TO BE THRASHED 
IT'S ALSO BASICALLY UNREADABLE
A BETTER IMPLEMENTATION WOULD NOT REQUIRE EVERY FUNCTION TO ACCESS MEMORY
*/
// Rho
__device__ float XFluxRho(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size]; // x-momentum density
}
__device__ float YFluxRho(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size]; // y-momentum density
}
__device__ float ZFluxRho(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size]; // z-momentum density
}

// RhoVX
__device__ float XFluxRhoVX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    float Bsq = B_sq(i, j, k, fluidvar, Nx, Ny, Nz);
    float ke = KE(i, j, k, fluidvar, Nx, Ny, Nz);
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size], 2)
        - pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size], 2) + p(i, j, k, fluidvar, Bsq, ke, Nx, Ny, Nz)
        + Bsq / 2.0;
}
__device__ float YFluxRhoVX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size]
        - fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
}
__device__ float ZFluxRhoVX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size]
        - fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
}

// RhoVY
__device__ float XFluxRhoVY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    // int cube_size = Nx * Ny * Nz;
    // return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size]
    //     - fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
    return YFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz);
}
__device__ float YFluxRhoVY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    float Bsq = B_sq(i, j, k, fluidvar, Nx, Ny, Nz);
    float ke = KE(i, j, k, fluidvar, Nx, Ny, Nz);
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size], 2)
        - pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size], 2) + p(i, j, k, fluidvar, Bsq, ke, Nx, Ny, Nz)
        + Bsq / 2.0;
}
__device__ float ZFluxRhoVY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size]
        - fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
}

// RhoVZ
__device__ float XFluxRhoVZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return ZFluxRhoVX(i, j, k, fluidvar, Nx, Ny, Nz);
}
__device__ float YFluxRhoVZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return ZFluxRhoVY(i, j, k, fluidvar, Nx, Ny, Nz);
}
__device__ float ZFluxRhoVZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    float Bsq = B_sq(i, j, k, fluidvar, Nx, Ny, Nz);
    float ke = KE(i, j, k, fluidvar, Nx, Ny, Nz);
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size], 2)
        - pow(fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size], 2) + p(i, j, k, fluidvar, Bsq, ke, Nx, Ny, Nz)
        + Bsq / 2.0;
}

// Bx
__device__ float XFluxBX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return 0.0;
}
__device__ float YFluxBX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size]
        - fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size];
}
__device__ float ZFluxBX(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size]
        - fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
}

// By
__device__ float XFluxBY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return -1.0 * YFluxBX(i, j, k, fluidvar, Nx, Ny, Nz);
}
__device__ float YFluxBY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return 0.0;
}
__device__ float ZFluxBY(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    int cube_size = Nx * Ny * Nz;
    return (1.0 / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)]) * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size]
        - fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size] * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size];
}

// Bz
__device__ float XFluxBZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return -1.0 * ZFluxBX(i, j, k, fluidvar, Nx, Ny, Nz);
}
__device__ float YFluxBZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return -1.0 * ZFluxBY(i, j, k, fluidvar, Nx, Ny, Nz);
}
__device__ float ZFluxBZ(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    return 0.0;
}

// e
__device__ float XFluxE(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    float Bsq = B_sq(i, j, k, fluidvar, Nx, Ny, Nz);
    float ke = KE(i, j, k, fluidvar, Nx, Ny, Nz);
    float pressure = p(i, j, k, fluidvar, Bsq, ke, Nx, Ny, Nz);
    float Bdotu = B_dot_u(i, j, k, fluidvar, Nx, Ny, Nz);
    int cube_size = Nx * Ny * Nz;
    return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + pressure + Bsq 
        * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + cube_size] / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)])
        - Bdotu * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 4 * cube_size];
}
__device__ float YFluxE(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    float Bsq = B_sq(i, j, k, fluidvar, Nx, Ny, Nz);
    float ke = KE(i, j, k, fluidvar, Nx, Ny, Nz);
    float pressure = p(i, j, k, fluidvar, Bsq, ke, Nx, Ny, Nz);
    float Bdotu = B_dot_u(i, j, k, fluidvar, Nx, Ny, Nz);
    int cube_size = Nx * Ny * Nz;
    return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + pressure + Bsq 
        * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 2 * cube_size] / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)])
        - Bdotu * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 5 * cube_size];
}
__device__ float ZFluxE(const int i, const int j, const int k, const float* fluidvar, const int Nx, const int Ny, const int Nz)
{
    float Bsq = B_sq(i, j, k, fluidvar, Nx, Ny, Nz);
    float ke = KE(i, j, k, fluidvar, Nx, Ny, Nz);
    float pressure = p(i, j, k, fluidvar, Bsq, ke, Nx, Ny, Nz);
    float Bdotu = B_dot_u(i, j, k, fluidvar, Nx, Ny, Nz);
    int cube_size = Nx * Ny * Nz;
    return fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 7 * cube_size] + pressure + Bsq 
        * (fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 3 * cube_size] / fluidvar[IDX3D(i, j, k, Nx, Ny, Nz)])
        - Bdotu * fluidvar[IDX3D(i, j, k, Nx, Ny, Nz) + 6 * cube_size];
}