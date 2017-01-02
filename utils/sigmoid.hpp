#ifndef NNDL_UTILS_SIGMOID_HPP
#define NNDL_UTILS_SIGMOID_HPP

#include <utils/types.hpp>

double sigmoid(double z);
vector sigmoid(const vector& z);

double dsigmoid(double sigmoid_z);
vector dsigmoid(const vector& sigmoid_z);

#endif  // NNDL_UTILS_SIGMOID_HPP
