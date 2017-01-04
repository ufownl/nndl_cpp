#ifndef NNDL_UTILS_MNIST_EVALUATOR_HPP
#define NNDL_UTILS_MNIST_EVALUATOR_HPP

#include <utils/types.hpp>
#include <iostream>
#include <chrono>

template <class T>
class mnist_evaluator {
public:
  mnist_evaluator(const data_set& test_data)
    : test_data_(test_data)
    , prev_(std::chrono::high_resolution_clock::now()) {
    // nop
  }

  void operator()(const T& nn, uint32_t epoch) {
    auto now = std::chrono::high_resolution_clock::now();
    auto result = 0u;
    for (auto i = test_data_.begin(); i != test_data_.end(); ++i) {
      auto out = nn.feedforward(i->first);
      if (mnist_decode_result(out) == mnist_decode_result(i->second)) {
        ++result;
      }
    }
    auto dur
      = std::chrono::duration_cast<std::chrono::milliseconds>(now - prev_);
    std::cerr << "Epoch " << epoch << ": "
              << result << " / " << test_data_.size() << "\t"
              << dur.count() << " ms" << std::endl;
    prev_ = std::chrono::high_resolution_clock::now();
  }

private:
  static uint8_t mnist_decode_result(const vector& x) {
    auto max_idx = 0;
    x.maxCoeff(&max_idx);
    return max_idx;
  }

  const data_set& test_data_;
  std::chrono::time_point<std::chrono::high_resolution_clock> prev_;
};

#endif  // NNDL_UTILS_MNIST_EVALUATOR_HPP
