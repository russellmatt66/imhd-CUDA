#ifndef FLUX_FUNC_HPP
#define FLUX_FUNC_HPP

#include "idealMHDFluid.hpp"

#include <type_traits>

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
XFluxRhoU(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);
template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
YFluxRhoU(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);
template <typename T> 
typename std::conditional<std::is_same<T, float>::value, float, double>::type 
ZFluxRhoU(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
XFluxRhoV(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);
template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
YFluxRhoV(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);
template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
ZFluxRhoV(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
XFluxRhoW(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);
template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
YFluxRhoW(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);
template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
ZFluxRhoW(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
YFluxBx(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);
template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
ZFluxBx(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
XFluxBy(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);
template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
ZFluxBy(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
XFluxBz(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);
template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
YFluxBz(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
XFluxE(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);
template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
YFluxE(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);
template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
ZFluxE(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);
#endif