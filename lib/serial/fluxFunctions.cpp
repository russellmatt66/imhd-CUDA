#include "fluxFunctions.hpp"
#include "helper.hpp"

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
XFluxRhoU(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return pow(imhd_fluid.rhovx(i, j, k), 2) / imhd_fluid.rho(i, j, k) - pow(imhd_fluid.Bx(i, j, k), 2) + pressure(imhd_fluid, i, j, k) + 0.5 * Bsq(imhd_fluid, i, j, k);
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
XFluxRhoV(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (1.0 / imhd_fluid.rho(i, j, k)) * (imhd_fluid.rhovy(i, j, k) * imhd_fluid.rhovx(i, j, k)) - imhd_fluid.By(i, j, k) * imhd_fluid.Bx(i, j, k);
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
XFluxRhoW(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (1.0 / imhd_fluid.rho(i, j, k)) * (imhd_fluid.rhovz(i, j, k) * imhd_fluid.rhovx(i, j, k)) - imhd_fluid.Bz(i, j, k) * imhd_fluid.Bx(i, j, k);
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
XFluxBy(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (1.0 / imhd_fluid.rho(i, j, k)) * (imhd_fluid.rhovy(i, j, k) * imhd_fluid.Bx(i, j, k) - imhd_fluid.By(i, j, k) * imhd_fluid.rhovx(i, j, k));
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
XFluxBz(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (1.0 / imhd_fluid.rho(i, j, k)) * (imhd_fluid.rhovz(i, j, k) * imhd_fluid.Bx(i, j, k) - imhd_fluid.Bz(i, j, k) * imhd_fluid.rhovx(i, j, k));
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
XFluxE(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (imhd_fluid.rhovx(i, j, k) / imhd_fluid.rho(i, j, k)) * (imhd_fluid.e(i, j, k) 
        + pressure(imhd_fluid, i, j, k) + 0.5 * Bsq(imhd_fluid, i, j, k)) 
        - imhd_fluid.Bx(i, j, k) * BdotU(imhd_fluid, i, j, k);
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
YFluxRhoU(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (1.0 / imhd_fluid.rho(i, j, k)) * (imhd_fluid.rhovx(i, j, k) * imhd_fluid.rhovy(i, j, k)) - imhd_fluid.Bx(i, j, k) * imhd_fluid.By(i, j, k);
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
YFluxRhoV(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return pow(imhd_fluid.rhovy(i, j, k), 2) / imhd_fluid.rho(i, j, k) - pow(imhd_fluid.By(i, j, k), 2) + pressure(imhd_fluid, i, j, k) + 0.5 * Bsq(imhd_fluid, i, j, k);
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
YFluxRhoW(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (1.0 / imhd_fluid.rho(i, j, k)) * (imhd_fluid.rhovz(i, j, k) * imhd_fluid.rhovy(i, j, k)) - imhd_fluid.Bz(i, j, k) * imhd_fluid.By(i, j, k);
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
YFluxBx(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (imhd_fluid.rhovx(i, j, k) / imhd_fluid.rho(i, j, k)) * imhd_fluid.By(i, j, k) - (imhd_fluid.rhovy(i, j, k) / imhd_fluid.rho(i, j, k)) * imhd_fluid.Bx(i, j, k);
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
YFluxBz(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (imhd_fluid.rhovz(i, j, k) / imhd_fluid.rho(i, j, k)) * imhd_fluid.By(i, j, k) - (imhd_fluid.rhovy(i, j, k) / imhd_fluid.rho(i, j, k)) * imhd_fluid.Bz(i, j, k);
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
YFluxE(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (imhd_fluid.rhovy(i, j, k) / imhd_fluid.rho(i, j, k)) * (imhd_fluid.e(i, j, k) 
        + pressure(imhd_fluid, i, j, k) + 0.5 * Bsq(imhd_fluid, i, j, k)) 
        - imhd_fluid.By(i, j, k) * BdotU(imhd_fluid, i, j, k);
}

template <typename T> 
typename std::conditional<std::is_same<T, float>::value, float, double>::type 
ZFluxRhoU(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (1.0 / imhd_fluid.rho(i, j, k)) * (imhd_fluid.rhovx(i, j, k) * imhd_fluid.rhovz(i, j, k)) - imhd_fluid.Bx(i, j, k) * imhd_fluid.Bz(i, j, k);
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
ZFluxRhoV(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (1.0 / imhd_fluid.rho(i, j, k)) * (imhd_fluid.rhovy(i, j, k) * imhd_fluid.rhovz(i, j, k)) - imhd_fluid.By(i, j, k) * imhd_fluid.Bz(i, j, k);
}

template <typename T> 
typename std::conditional<std::is_same<T, float>::value, float, double>::type 
ZFluxRhoW(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return pow(imhd_fluid.rhovz(i, j, k), 2) / imhd_fluid.rho(i, j, k) - pow(imhd_fluid.Bz(i, j, k), 2) + pressure(imhd_fluid, i, j, k) + 0.5 * Bsq(imhd_fluid, i, j, k);
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
ZFluxBx(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (1.0 / imhd_fluid.rho(i, j, k)) * (imhd_fluid.rhovx(i, j, k) * imhd_fluid.Bz(i, j, k) - imhd_fluid.rhovz(i, j, k) - imhd_fluid.Bx(i, j, k));
}

template <typename T> 
typename std::conditional<std::is_same<T, float>::value, float, double>::type 
ZFluxBy(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
    return (1.0 / imhd_fluid.rho(i, j, k)) * (imhd_fluid.rhovy(i, j, k) * imhd_fluid.Bz(i, j, k) - imhd_fluid.rhovz(i, j, k) - imhd_fluid.By(i, j, k));
}

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
ZFluxE(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k){
        return (imhd_fluid.rhovz(i, j, k) / imhd_fluid.rho(i, j, k)) * (imhd_fluid.e(i, j, k) 
        + pressure(imhd_fluid, i, j, k) + 0.5 * Bsq(imhd_fluid, i, j, k)) 
        - imhd_fluid.Bz(i, j, k) * BdotU(imhd_fluid, i, j, k);
}