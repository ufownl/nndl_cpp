#include "mnist_evaluator.hpp"

uint8_t mnist_decode_result(const vector& x) {
  auto max_idx = 0;
  x.maxCoeff(&max_idx);
  return max_idx;
}
