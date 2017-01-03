#ifndef NNDL_CHAPTER1_NEURAL_NETWORK_HPP
#define NNDL_CHAPTER1_NEURAL_NETWORK_HPP

#include <utils/types.hpp>
#include <functional>
#include <random>

class neural_network {
public:
  using evaluator =
    std::function<void(const neural_network& nn, uint32_t epoch)>;

  neural_network(std::vector<uint32_t> sizes);

  vector feedforward(vector a) const;
  void sgd_train(data_set& training_data, uint32_t epochs,
                 uint32_t mini_batch_size, double eta,
                 evaluator f = evaluator());

private:
  void update_mini_batch(data_set::const_iterator it0,
                         data_set::const_iterator it1, double eta);
  std::pair<std::vector<matrix>, std::vector<matrix>>
  backprop(matrix activation, const matrix& label);

  std::random_device rd_;
  std::mt19937 gen_ {rd_()};
  std::vector<uint32_t> sizes_;
  std::vector<vector> biases_;
  std::vector<matrix> weights_;
};

#endif  // NNDL_CHAPTER1_NEURAL_NETWORK_HPP