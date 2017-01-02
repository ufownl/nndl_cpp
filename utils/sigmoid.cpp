#include "sigmoid.hpp"
#include <math.h>
#include <assert.h>

double sigmoid(double z) {
  return 1.0 / (1.0 + exp(-z));
}

vector sigmoid(const vector& z) {
  vector r(z.rows(), 1);
  for (auto i = 0; i < z.rows(); ++i) {
    r(i) = sigmoid(z(i));
  }
  return r;
}

double dsigmoid(double sigmoid_z) {
  return sigmoid_z * (1.0 - sigmoid_z);
}

vector dsigmoid(const vector& sigmoid_z) {
  vector r(sigmoid_z.rows(), 1);
  for (auto i = 0; i < sigmoid_z.rows(); ++i) {
    r(i) = dsigmoid(sigmoid_z(i));
  }
  return r;
}
