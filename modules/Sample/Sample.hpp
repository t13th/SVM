#ifndef __SVM_SAMPLE_HPP__
#define __SVM_SAMPLE_HPP__

#include <array>
#include <cstddef>
namespace SVM {

enum ClassificationType { Positive = 1, Negative = -1 };

template <std::size_t Dimension, std::floating_point svm_float_t = double>
class Sample {
  ClassificationType type;
  std::array<svm_float_t, Dimension> data;

 public:
  svm_float_t& operator[](int) const;
  ClassificationType& get_type();
  std::array<svm_float_t, Dimension>& get_data();
};

}  // namespace SVM

//////////implementation//////////

namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t>
svm_float_t& Sample<Dimension, svm_float_t>::operator[](int index) const {
  return data[index];
}
template <std::size_t Dimension, std::floating_point svm_float_t>
std::array<svm_float_t, Dimension>& Sample<Dimension, svm_float_t>::get_data() {
  return data;
}
template <std::size_t Dimension, std::floating_point svm_float_t>
ClassificationType& Sample<Dimension, svm_float_t>::get_type() {
  return type;
}

}  // namespace SVM

#endif