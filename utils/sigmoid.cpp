#include "sigmoid.hpp"
#include <math.h>

double sigmoid(double z) {
  return 1.0 / (1.0 + exp(-z));
}

double dsigmoid(double sigmoid_z) {
  return sigmoid_z * (1.0 - sigmoid_z);
}
