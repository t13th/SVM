#ifndef __SVM_SVM_HPP__
#define __SVM_SVM_HPP__

#include <cstddef>

#include "../SegmentPlane/SegmentPlane.hpp"
#include "../common/common.hpp"

namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t = double>
struct SVM {
  SegmentPlane<Dimension, svm_float_t> segmentation;

  ClassificationType operator()(const FixedVector<Dimension, svm_float_t>&);
};
}  // namespace SVM

//////////Implementation//////////

namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t>
ClassificationType SVM<Dimension, svm_float_t>::operator()(
    const FixedVector<Dimension, svm_float_t>& data) {
  auto&& classfication = segmentation.weight * data + segmentation.bias;
  return static_cast<ClassificationType>(sgn(classfication) + 1);
}

}  // namespace SVM
#endif