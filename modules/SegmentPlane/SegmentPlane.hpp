#ifndef __SVM_SEGMENT_PLANE_HPP__
#define __SVM_SEGMENT_PLANE_HPP__

#include <array>
#include <cstddef>
namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t = double>
class SegmentPlane {
  std::array<svm_float_t, Dimension> weight;
  svm_float_t bias;

 public:
  svm_float_t& operator[](int) const;
  std::array<svm_float_t, Dimension>& get_weight();
  svm_float_t& get_bias();
};

}  // namespace SVM

//////////implementation//////////

namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t>
svm_float_t& SegmentPlane<Dimension, svm_float_t>::operator[](int index) const {
  return weight[index];
}
template <std::size_t Dimension, std::floating_point svm_float_t>
std::array<svm_float_t, Dimension>&
SegmentPlane<Dimension, svm_float_t>::get_weight() {
  return weight;
}
template <std::size_t Dimension, std::floating_point svm_float_t>
svm_float_t& SegmentPlane<Dimension, svm_float_t>::get_bias() {
  return bias;
}

}  // namespace SVM

#endif