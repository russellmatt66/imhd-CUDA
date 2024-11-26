#ifndef RANK_3_TENS
#define RANK_3_TENS

#include <stddef.h>

template <typename T>
class rank3Tensor { // High-performance implementation of a rank 3 tensor
    public:
        rank3Tensor(size_t Nx, size_t Ny, size_t Nz) : num_rows_(Nx), num_cols_(Ny), num_layers_(Nz_), storage_(num_rows_ * num_cols_ * num_layers_, 0.0) {}

        // row-major, depth-minor
        T& operator()(size_t i, size_t j, size_t k) { return storage_[k * num_rows_ * num_cols_ + i * num_cols_ + j]; }
        const T& operator()(size_t i, size_t j, size_t k) const { return storage_[k * num_rows_ * num_cols_ + i * num_cols_ + j]; }
        
        const std::vector<T>& get_storage() const { return storage_; }

        const size_t num_rows() const { return num_rows_; }
        const size_t num_cols() const { return num_cols_; }
        const size_t num_layers() const { return num_layers_; }

    private:
        size_t num_rows_, num_cols_, num_layers_;
        std::vector<T> storage_;
};

#endif