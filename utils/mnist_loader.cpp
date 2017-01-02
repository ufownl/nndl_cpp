#include "mnist_loader.hpp"

namespace {

uint32_t mnist_load_uint32(std::ifstream& in) {
  uint32_t result = 0;
  for (auto i = 0; i < 4; ++i) {
    uint8_t byte;
    in.read(reinterpret_cast<char*>(&byte), 1);
    result |= byte << (3 - i) * 8;
  }
  return result;
}

}

mnist_images mnist_load_images(std::ifstream& in) {
  auto magic = mnist_load_uint32(in);
  if (magic != 2051) {
    return {};
  }
  auto count = mnist_load_uint32(in);
  mnist_images images;
  images.reserve(count);
  auto rows = mnist_load_uint32(in);
  auto cols = mnist_load_uint32(in);
  for (auto i = 0; i < count; ++i) {
    auto size = rows * cols;
    vector img(size, 1);
    for (auto j = 0; j < size; ++j) {
      uint8_t byte;
      in.read(reinterpret_cast<char*>(&byte), 1);
      img(j) = byte / 255.0;
    }
    images.emplace_back(std::move(img));
  }
  return images;
}

mnist_labels mnist_load_labels(std::ifstream& in) {
  auto magic = mnist_load_uint32(in);
  if (magic != 2049) {
    return {};
  }
  auto count = mnist_load_uint32(in);
  mnist_labels labels(count);
  for (auto& l: labels) {
    in.read(reinterpret_cast<char*>(&l), 1);
  }
  return labels;
}

data_set mnist_data_set(const mnist_images& images,
                        const mnist_labels& labels) {
  auto count = std::min(images.size(), labels.size());
  data_set data;
  data.reserve(count);
  for (auto i = 0; i < count; ++i) {
    vector label(10, 1);
    for (auto j = 0; j < 10; ++j) {
      label(j) = j == labels[i] ? 1.0 : 0.0;
    }
    data.emplace_back(images[i], std::move(label));
  }
  return data;
}

uint8_t mnist_decode_result(const vector& x) {
  auto max_val = 0.0;
  auto max_idx = 0;
  for (auto i = 0; i < x.rows(); ++i) {
    if (x(i) > max_val) {
      max_val = x(i);
      max_idx = i;
    }
  }
  return max_idx;
}
