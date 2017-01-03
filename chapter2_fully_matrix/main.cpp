#include "neural_network.hpp"
#include <utils/mnist_loader.hpp>
#include <iostream>

namespace {

mnist_images load_images() {
  std::ifstream in("../../data/train-images-idx3-ubyte", std::ios::binary);
  return mnist_load_images(in);
}

mnist_labels load_labels() {
  std::ifstream in("../../data/train-labels-idx1-ubyte", std::ios::binary);
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

}

int main() {
  std::cerr << "Loading MNIST data..." << std::endl;
  auto data = make_data_set(load_images(), load_labels());
  auto training_data = std::move(data.first);
  auto test_data = std::move(data.second);
  std::cerr << "Complete!" << std::endl;
  neural_network nn({784u, 30u, 10u});
  nn.sgd_train(training_data, 30u, 10u, 3.0,
    [&test_data](const neural_network& self, uint32_t epoch) {
      auto result = 0u;
      for (auto i = test_data.begin(); i != test_data.end(); ++i) {
        auto out = self.feedforward(i->first);
        if (mnist_decode_result(out) == mnist_decode_result(i->second)) {
          ++result;
        }
      }
      std::cerr << "Epoch " << epoch << ": "
                << result << " / " << test_data.size() << std::endl;
    }
  );
  return 0;
}
