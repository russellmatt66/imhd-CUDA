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
            rho_(Nx, Ny, Nz),
            rhovx_(Nx, Ny, Nz),
            rhovy_(Nx, Ny, Nz),
            rhovz_(Nx, Ny, Nz),
            Bx_(Nx, Ny, Nz),
            By_(Nx, Ny, Nz),
            Bz_(Nx, Ny, Nz),
            e_(Nx, Ny, Nz)
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

// I want to compare speed b/w storing fluxes in memory and computing them on the fly 
template <typename T>
class idealMHDFluxes{
    public:
        idealMHDFluxes(size_t Nx, size_t Ny, size_t Nz) : 
            num_rows_(Nx),
            num_cols_(Ny),
            num_layers_(Nz),
            xflux_rhou_(Nx, Ny, Nz),
            yflux_rhou_(Nx, Ny, Nz),
            zflux_rhou_(Nx, Ny, Nz),
            xflux_rhov_(Nx, Ny, Nz),
            yflux_rhov_(Nx, Ny, Nz),
            zflux_rhov_(Nx, Ny, Nz),
            xflux_rhow_(Nx, Ny, Nz),
            yflux_rhow_(Nx, Ny, Nz),
            zflux_rhow_(Nx, Ny, Nz),
            yflux_Bx_(Nx, Ny, Nz),
            zflux_Bx_(Nx, Ny, Nz),
            xflux_By_(Nx, Ny, Nz),
            zflux_By_(Nx, Ny, Nz),
            xflux_Bz_(Nx, Ny, Nz),
            yflux_Bz_(Nx, Ny, Nz),
            xflux_e_(Nx, Ny, Nz),
            yflux_e_(Nx, Ny, Nz),
            zflux_e_(Nx, Ny, Nz) 
            {}

        // Accessors
        T& xflux_rhou(size_t i, size_t j, size_t k) { return xflux_rhou_(i, j, k); }
        T& yflux_rhou(size_t i, size_t j, size_t k) { return yflux_rhou_(i, j, k); }
        T& zflux_rhou(size_t i, size_t j, size_t k) { return zflux_rhou_(i, j, k); }
        const T& xflux_rhou(size_t i, size_t j, size_t k) const { return xflux_rhou_(i, j, k); }
        const T& yflux_rhou(size_t i, size_t j, size_t k) const { return yflux_rhou_(i, j, k); }
        const T& zflux_rhou(size_t i, size_t j, size_t k) const { return zflux_rhou_(i, j, k); }

        T& xflux_rhov(size_t i, size_t j, size_t k) { return xflux_rhov_(i, j, k); }
        T& yflux_rhov(size_t i, size_t j, size_t k) { return yflux_rhov_(i, j, k); }
        T& zflux_rhov(size_t i, size_t j, size_t k) { return zflux_rhov_(i, j, k); }
        const T& xflux_rhov(size_t i, size_t j, size_t k) const { return xflux_rhov_(i, j, k); }
        const T& yflux_rhov(size_t i, size_t j, size_t k) const { return yflux_rhov_(i, j, k); }
        const T& zflux_rhov(size_t i, size_t j, size_t k) const { return zflux_rhov_(i, j, k); }

        T& xflux_rhow(size_t i, size_t j, size_t k) { return xflux_rhow_(i, j, k); }
        T& yflux_rhow(size_t i, size_t j, size_t k) { return yflux_rhow_(i, j, k); }
        T& zflux_rhow(size_t i, size_t j, size_t k) { return zflux_rhow_(i, j, k); }
        const T& xflux_rhow(size_t i, size_t j, size_t k) const { return xflux_rhow_(i, j, k); }
        const T& yflux_rhow(size_t i, size_t j, size_t k) const { return yflux_rhow_(i, j, k); }
        const T& zflux_rhow(size_t i, size_t j, size_t k) const { return zflux_rhow_(i, j, k); }

        T& yflux_Bx(size_t i, size_t j, size_t k) { return yflux_Bx_(i, j, k); }
        T& zflux_Bx(size_t i, size_t j, size_t k) { return zflux_Bx_(i, j, k); }
        const T& yflux_Bx(size_t i, size_t j, size_t k) const { return yflux_Bx_(i, j, k); }
        const T& zflux_Bx(size_t i, size_t j, size_t k) const { return zflux_Bx_(i, j, k); }

        T& xflux_By(size_t i, size_t j, size_t k) { return xflux_By_(i, j, k); }
        T& zflux_By(size_t i, size_t j, size_t k) { return zflux_By_(i, j, k); }
        const T& xflux_By(size_t i, size_t j, size_t k) const { return xflux_By_(i, j, k); }
        const T& zflux_By(size_t i, size_t j, size_t k) const { return zflux_By_(i, j, k); }

        T& xflux_Bz(size_t i, size_t j, size_t k) { return xflux_Bz_(i, j, k); }
        T& yflux_Bz(size_t i, size_t j, size_t k) { return yflux_Bz_(i, j, k); }
        const T& xflux_Bz(size_t i, size_t j, size_t k) const { return xflux_Bz_(i, j, k); }
        const T& yflux_Bz(size_t i, size_t j, size_t k) const { return yflux_Bz_(i, j, k); }

        T& xflux_e(size_t i, size_t j, size_t k) { return xflux_e_(i, j, k); }
        T& yflux_e(size_t i, size_t j, size_t k) { return yflux_e_(i, j, k); }
        T& zflux_e(size_t i, size_t j, size_t k) { return zflux_e_(i, j, k); }
        const T& xflux_e(size_t i, size_t j, size_t k) const { return xflux_e_(i, j, k); }
        const T& yflux_e(size_t i, size_t j, size_t k) const { return yflux_e_(i, j, k); }
        const T& zflux_e(size_t i, size_t j, size_t k) const { return zflux_e_(i, j, k); }

    private:
        size_t num_rows_, num_cols_, num_layers_;
        rank3Tensor<T> xflux_rhou_; // Don't need to store rho-fluxes b/c they are just the momentum densities
        rank3Tensor<T> yflux_rhou_;
        rank3Tensor<T> zflux_rhou_;

        rank3Tensor<T> xflux_rhov_;
        rank3Tensor<T> yflux_rhov_;
        rank3Tensor<T> zflux_rhov_;

        rank3Tensor<T> xflux_rhow_;
        rank3Tensor<T> yflux_rhow_;
        rank3Tensor<T> zflux_rhow_;

        rank3Tensor<T> yflux_Bx_; // div B = 0
        rank3Tensor<T> zflux_Bx_;

        rank3Tensor<T> xflux_By_;
        rank3Tensor<T> zflux_By_;

        rank3Tensor<T> xflux_Bz_;
        rank3Tensor<T> yflux_Bz_;

        rank3Tensor<T> xflux_e_;
        rank3Tensor<T> yflux_e_;
        rank3Tensor<T> zflux_e_;
};
#endif