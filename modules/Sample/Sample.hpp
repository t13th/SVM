#ifndef __SVM_SAMPLE_HPP__
#define __SVM_SAMPLE_HPP__

#include <cstddef>

#include "common/common.hpp"

namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t = double>
struct Sample {
  ClassificationType classification;
  FixedVector<Dimension, svm_float_t> data;

  svm_float_t& operator[](int index) { return data[index]; }
};

}  // namespace SVM

#endif