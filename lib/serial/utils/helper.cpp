#include "helper.hpp"
#include "idealMHDFluid.hpp"

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
KE(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (0.5 / imhd_fluid.rho(i, j, k)) * (pow(imhd_fluid.rhovx(i, j, k), 2) + pow(imhd_fluid.rhovy(i, j, k), 2) + pow(imhd_fluid.rhovz(i, j, k), 2));
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
Bsq(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (pow(imhd_fluid.Bx(i, j, k), 2) + pow(imhd_fluid.By(i, j, k), 2) + pow(imhd_fluid.Bz(i, j, k), 2));
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
pressure(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (imhd_fluid.getGamma() - 1.0) * (imhd_fluid.e(i, j, k) - KE(imhd_fluid, i, j, k) - 0.5 * Bsq(imhd_fluid, i, j, k));
}   

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
BdotU(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (1.0 / imhd_fluid.rho(i, j, k)) * (imhd_fluid.Bx(i, j, k) * imhd_fluid.rhovx(i, j, k) 
        + imhd_fluid.By(i, j, k) * imhd_fluid.rhovy(i, j, k) + imhd_fluid.Bz(i, j, k) * imhd_fluid.rhovz(i, j, k));
}