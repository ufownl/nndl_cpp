#ifndef NNDL_UTILS_MNIST_EVALUATOR_HPP
#define NNDL_UTILS_MNIST_EVALUATOR_HPP

#include <utils/types.hpp>
#include <iostream>
#include <chrono>

template <class T, class R = void>
class mnist_evaluator {
public:
  mnist_evaluator(const data_set& test_data)
    : test_data_(test_data)
    , prev_(std::chrono::high_resolution_clock::now()) {
    // nop
  }

  R operator()(const T& nn, uint32_t epoch) {
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
    return R();
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

template <class T>
class mnist_early_stopping {
public:
  mnist_early_stopping(const data_set& test_data, uint32_t n)
    : test_data_(test_data)
    , stop_n_(n)
    , prev_(std::chrono::high_resolution_clock::now()) {
    // nop
  }

  bool operator()(const T& nn, uint32_t epoch) {
    auto now = std::chrono::high_resolution_clock::now();
    auto result = 0u;
    for (auto i = test_data_.begin(); i != test_data_.end(); ++i) {
      auto out = nn.feedforward(i->first);
      if (mnist_decode_result(out) == mnist_decode_result(i->second)) {
        ++result;
      }
    }
    if (result > max_result_) {
      max_result_ = result;
      max_epoch_ = epoch;
    }
    auto dur
      = std::chrono::duration_cast<std::chrono::milliseconds>(now - prev_);
    std::cerr << "Epoch " << epoch << ": "
              << result << " / " << test_data_.size() << "\t"
              << dur.count() << " ms" << std::endl;
    prev_ = std::chrono::high_resolution_clock::now();
    return epoch - max_epoch_ >= stop_n_;
  }

private:
  static uint8_t mnist_decode_result(const vector& x) {
    auto max_idx = 0;
    x.maxCoeff(&max_idx);
    return max_idx;
  }

  const data_set& test_data_;
  uint32_t stop_n_;
  uint32_t max_result_ {0};
  uint32_t max_epoch_;
  std::chrono::time_point<std::chrono::high_resolution_clock> prev_;
};

#endif  // NNDL_UTILS_MNIST_EVALUATOR_HPP
