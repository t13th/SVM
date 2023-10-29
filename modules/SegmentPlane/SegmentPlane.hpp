#ifndef __SVM_SEGMENT_PLANE_HPP__
#define __SVM_SEGMENT_PLANE_HPP__

#include <cstddef>

#include "../common/common.hpp"
namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t = double>
struct SegmentPlane {
  FixedVector<Dimension, svm_float_t> weight;
  svm_float_t bias;

  svm_float_t& operator[](int index) { return weight[index]; }
};

}  // namespace SVM

#endif