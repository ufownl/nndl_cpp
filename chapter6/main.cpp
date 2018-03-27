#include "loss_multiattr.hpp"
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <iostream>
#include <vector>

using input_layer =
  dlib::input<dlib::matrix<unsigned char>>;

using conv_pool_layer0 =
  dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<6, 5, 5, 1, 1,
                                                  input_layer>>>;

using conv_pool_layer1 =
  dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<16, 5, 5, 1, 1,
                                                  conv_pool_layer0>>>;

using fully_connected_layer0 =
  dlib::relu<dlib::fc<120, conv_pool_layer1>>;

using fully_connected_layer1 =
  dlib::relu<dlib::fc<84, fully_connected_layer0>>;

using neural_network =
  loss_multiattr<dlib::sig<dlib::fc<10, fully_connected_layer1>>>;

int main() {
  std::cerr << "Loading MNIST data..." << std::endl;
  std::vector<dlib::matrix<unsigned char>> training_images;
  std::vector<unsigned long> training_labels;
  std::vector<dlib::matrix<unsigned char>> validation_images;
  std::vector<unsigned long> validation_labels;
  dlib::load_mnist_dataset("../../data", training_images, training_labels,
                           validation_images, validation_labels);
  std::vector<dlib::matrix<float>> labels;
  for (auto& label: training_labels) {
    dlib::matrix<float> mat(10, 1);
    for (auto i = 0ul; i < 10; ++i) {
      if (i == label) {
        mat(i, 0) = 1.0f;
      } else {
        mat(i, 0) = 0.0f;
      }
    }
    labels.emplace_back(std::move(mat));
  }
  std::cerr << "Complete!" << std::endl;
  neural_network nn;
  dlib::dnn_trainer<neural_network> trainer(nn);
  trainer.set_max_num_epochs(60);
  trainer.set_mini_batch_size(100);
  trainer.set_learning_rate(0.03);
  trainer.set_learning_rate_shrink_factor(0.5);
  trainer.be_verbose();
  trainer.train(training_images, labels);
  auto predicted_labels = nn(validation_images);
  auto num_right = 0u;
  auto num_wrong = 0u;
  for (auto i = 0; i < validation_images.size(); ++i) {
    if (dlib::index_of_max(dlib::colm(predicted_labels[i], 0)) == validation_labels[i]) {
      ++num_right;
    } else {
      ++num_wrong;
    }
  }
  std::cerr << "validation num_right: " << num_right << std::endl;
  std::cerr << "validation num_wrong: " << num_wrong << std::endl;
  std::cerr << "validation accuracy:  "
            << static_cast<double>(num_right)/(num_right+num_wrong)
            << std::endl;
  return 0;
}
