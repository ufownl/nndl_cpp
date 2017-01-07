#ifndef NNDL_CHAPTER3_NEURAL_NETWORK_HPP
#define NNDL_CHAPTER3_NEURAL_NETWORK_HPP

#include <utils/types.hpp>
#include <functional>
#include <random>

class neural_network {
public:
  using evaluator =
    std::function<bool(const neural_network&, uint32_t)>;

  neural_network(std::vector<uint32_t> sizes);

  vector feedforward(vector a) const;
  void sgd_train(data_set& training_data, uint32_t epochs,
                 uint32_t mini_batch_size, double eta, double lambda,
                 evaluator f = evaluator());

private:
  void update_mini_batch(data_set::const_iterator it0,
                         data_set::const_iterator it1, size_t total_size,
                         double eta, double lambda);
  std::pair<std::vector<vector>, std::vector<matrix>>
  backprop(matrix activation, const matrix& label);

  std::mt19937 rand_gen_;
  std::vector<uint32_t> sizes_;
  std::vector<vector> biases_;
  std::vector<matrix> weights_;
};

#endif  // NNDL_CHAPTER3_NEURAL_NETWORK_HPP
