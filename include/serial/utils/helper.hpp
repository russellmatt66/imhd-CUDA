#ifndef HELPER_HPP
#define HELPER_HPP

#include <type_traits>

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
pressure(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
KE(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
Bsq(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);

template <typename T>
typename std::conditional<std::is_same<T, float>::value, float, double>::type
BdotU(idealMHDFluid<T>& imhd_fluid, const size_t i, const size_t j, const size_t k);
#endif