#ifndef NNDL_UTILS_MNIST_LOADER_HPP
#define NNDL_UTILS_MNIST_LOADER_HPP

#include <utils/types.hpp>
#include <fstream>

using mnist_images = std::vector<vector>;
using mnist_labels = std::vector<uint8_t>;

mnist_images mnist_load_images(std::ifstream& in);
mnist_labels mnist_load_labels(std::ifstream& in);

data_set mnist_data_set(const mnist_images& images, const mnist_labels& labels);

#endif  // NNDL_UTILS_MNIST_LOADER_HPP
