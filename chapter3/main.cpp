#include "neural_network.hpp"
#include <utils/mnist_loader.hpp>
#include <utils/mnist_evaluator.hpp>
#include <limits>

namespace {

mnist_images load_images() {
  std::ifstream in("../../data/train-images-idx3-ubyte", std::ios::binary);
  return mnist_load_images(in);
}

mnist_labels load_labels() {
  std::ifstream in("../../data/train-labels-idx1-ubyte", std::ios::binary);
  return mnist_load_labels(in);
}

mnist_images load_validation_images() {
  std::ifstream in("../../data/t10k-images-idx3-ubyte", std::ios::binary);
  return mnist_load_images(in);
}

mnist_labels load_validation_labels() {
  std::ifstream in("../../data/t10k-labels-idx1-ubyte", std::ios::binary);
  return mnist_load_labels(in);
}

std::pair<data_set, data_set> make_data_set(const mnist_images& images,
                                            const mnist_labels& labels) {
  data_set training_data =
    mnist_data_set(mnist_images(images.begin(), images.begin() + 50000),
                   mnist_labels(labels.begin(), labels.begin() + 50000));
  data_set test_data =
    mnist_data_set(mnist_images(images.begin() + 50000, images.end()),
                   mnist_labels(labels.begin() + 50000, labels.end()));
  return {std::move(training_data), std::move(test_data)};
}

class learning_rate_scheduler {
public:
  learning_rate_scheduler(double eta) : eta_(eta) {
    // nop
  }

  double operator()() {
    auto res = eta_ / div_;
    div_ <<= 1u;
    return res;
  }

  uint32_t div() const {
    return div_;
  }

private:
  double eta_;
  uint32_t div_ = 1u;
};

}

int main() {
  std::cerr << "Loading MNIST data..." << std::endl;
  auto data = make_data_set(load_images(), load_labels());
  auto training_data = std::move(data.first);
  auto test_data = std::move(data.second);
  auto validation_data = mnist_data_set(load_validation_images(),
                                        load_validation_labels());
  std::cerr << "Complete!" << std::endl;
  neural_network nn({784u, 100u, 10u});
  for (learning_rate_scheduler eta(0.5); eta.div() <= 128u; ) {
    auto learning_rate = eta();
    std::cerr << "Learning rate: " << learning_rate << std::endl;
    nn.sgd_train(training_data, std::numeric_limits<uint32_t>::max(),
                 50u, learning_rate, 5.0, 0.8,
                 mnist_early_stopping<neural_network>(validation_data, 10u));
  }
  return 0;
}
