#ifndef NNDL_UTILS_TYPES_HPP
#define NNDL_UTILS_TYPES_HPP

#include <Eigen/Eigen>
#include <vector>
#include <utility>

using matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using data_set = std::vector<std::pair<vector, vector>>;

#endif  // NNDL_UTILS_TYPES_HPP
