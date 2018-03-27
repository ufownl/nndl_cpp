#ifndef AI_CHALLENGER_ZSL_LOSS_MULTIATTR_HPP
#define AI_CHALLENGER_ZSL_LOSS_MULTIATTR_HPP

#include <dlib/dnn/core.h>
#include <dlib/dnn/tensor_tools.h>
#include <dlib/matrix.h>
#include <iostream>
#include <string>

class loss_multiattr_ {
public:
  using training_label_type = dlib::matrix<float>;
  using output_label_type = dlib::matrix<float>;

  template <class Subnet, class LabelIterator>
  void to_label(const dlib::tensor& input_tensor,
                const Subnet& sub,
                LabelIterator iter) const {
    auto& output_tensor = sub.get_output();
    DLIB_CASSERT(sub.sample_expansion_factor() == 1);
    DLIB_CASSERT(output_tensor.nr() == 1 && output_tensor.nc() == 1);
    DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
    auto out_data = output_tensor.host();
    for (auto i = 0ll; i < output_tensor.num_samples(); ++i) {
      *iter++ = dlib::mat(out_data, output_tensor.k(), 1);
      out_data += output_tensor.k();
    }
  }

  template <class ConstLabelIterator, class Subnet>
  double compute_loss_value_and_gradient(const dlib::tensor& input_tensor,
                                         ConstLabelIterator truth, 
                                         Subnet& sub) const {
    auto& output_tensor = sub.get_output();
    auto& grad = sub.get_gradient_input();
    DLIB_CASSERT(sub.sample_expansion_factor() == 1);
    DLIB_CASSERT(input_tensor.num_samples() != 0);
    DLIB_CASSERT(input_tensor.num_samples() % sub.sample_expansion_factor() == 0);
    DLIB_CASSERT(input_tensor.num_samples() == grad.num_samples());
    DLIB_CASSERT(input_tensor.num_samples() == output_tensor.num_samples());
    DLIB_CASSERT(output_tensor.nr() == 1 && output_tensor.nc() == 1);
    DLIB_CASSERT(grad.nr() == 1 && grad.nc() == 1);
    // The loss we output is the average loss over the mini-batch.
    auto scale = 1.0f / output_tensor.num_samples();
    auto loss = 0.0;
    auto out_data = output_tensor.host();
    auto g = grad.host();
    for (auto i = 0ll; i < output_tensor.num_samples(); ++i) {
      auto& y = *truth++;
      for (auto k = 0ll; k < output_tensor.k(); ++k) {
        auto idx = i * output_tensor.k() + k;
        auto d = out_data[idx] - y(k, 0);
        loss += scale * d * d;
        g[idx] = scale * d * out_data[idx] * (1.0f - out_data[idx]);
      }
    }
    return loss;
  }

  friend void serialize(const loss_multiattr_& , std::ostream& out) {
    dlib::serialize("loss_multiattr_", out);
  }

  friend void deserialize(loss_multiattr_& , std::istream& in) {
    std::string version;
    dlib::deserialize(version, in);
    if (version != "loss_multiattr_") {
      throw dlib::serialization_error("Unexpected version found while deserializing loss_multiattr_.");
    }
  }

  friend std::ostream& operator<<(std::ostream& out, const loss_multiattr_&) {
    out << "loss_multiattr";
    return out;
  }

  friend void to_xml(const loss_multiattr_&, std::ostream& out) {
    out << "<loss_multiattr/>";
  }
};

template <typename Subnet>
using loss_multiattr = dlib::add_loss_layer<loss_multiattr_, Subnet>;

#endif  // AI_CHALLENGER_ZSL_LOSS_MULTIATTR_HPP
