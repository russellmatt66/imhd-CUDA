#ifndef IMHD_FLUID
#define IMHD_FLUID

#include "rank3Tensor.hpp"

template <typename T>
class idealMHDFluid{
    public:
        idealMHDFluid(size_t Nx, size_t Ny, size_t Nz) : 
            num_rows_(Nx),
            num_cols_(Ny),
            num_layers_(Nz),
            rank3Tensor<T> rho_(Nx, Ny, Nz),
            rank3Tensor<T> rhovx_(Nx, Ny, Nz),
            rank3Tensor<T> rhovy_(Nx, Ny, Nz),
            rank3Tensor<T> rhovz_(Nx, Ny, Nz),
            rank3Tensor<T> Bx_(Nx, Ny, Nz),
            rank3Tensor<T> By_(Nx, Ny, Nz),
            rank3Tensor<T> Bz_(Nx, Ny, Nz),
            rank3Tensor<T> e_(Nx, Ny, Nz)
            {}
        
        // Fluid variable accessors 
        T& rho(size_t i, size_t j, size_t k) { return rho_(i, j, k); }
        const T& rho(size_t i, size_t j, size_t k) const { return rho_(i, j, k); }

        T& rhovx(size_t i, size_t j, size_t k) { return rhovx_(i, j, k); }
        const T& rhovx(size_t i, size_t j, size_t k) const { return rhovx_(i, j, k); }
        
        T& rhovy(size_t i, size_t j, size_t k) { return rhovy_(i, j, k); }
        const T& rhovy(size_t i, size_t j, size_t k) const { return rhovy_(i, j, k); }

        T& rhovz(size_t i, size_t j, size_t k) { return rhovz_(i, j, k); }
        const T& rhovz(size_t i, size_t j, size_t k) const { return rhovz_(i, j, k); }
    
        T& Bx(size_t i, size_t j, size_t k) { return Bx_(i, j, k); }
        const T& Bx(size_t i, size_t j, size_t k) const { return Bx_(i, j, k); }

        T& By(size_t i, size_t j, size_t k) { return By_(i, j, k); }
        const T& By(size_t i, size_t j, size_t k) const { return By_(i, j, k); }

        T& Bz(size_t i, size_t j, size_t k) { return Bz_(i, j, k); }
        const T& Bz(size_t i, size_t j, size_t k) const { return Bz_(i, j, k); }
        
        T& e(size_t i, size_t j, size_t k) { return e_(i, j, k); }
        const T& e(size_t i, size_t j, size_t k) const { return e_(i, j, k); }
    
        // Physical constants
        // mu_0 = 1
        const T getGamma() const { return gamma_; } // polytropic index for an ideal gas

    private:
        size_t num_rows_, num_cols_, num_layers_;
        T gamma_ = 5.0 / 3.0; // polytropic index for an ideal gas
        rank3Tensor<T> rho_;
        rank3Tensor<T> rhovx_;
        rank3Tensor<T> rhovy_;
        rank3Tensor<T> rhovz_;
        rank3Tensor<T> Bx_;
        rank3Tensor<T> By_;
        rank3Tensor<T> Bz_;
        rank3Tensor<T> e_;
};

#endif