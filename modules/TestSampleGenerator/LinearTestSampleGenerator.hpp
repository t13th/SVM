#ifndef __SVM_LINEAR_TEST_SAMPLE_GENERATOR_HPP__
#define __SVM_LINEAR_TEST_SAMPLE_GENERATOR_HPP__

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>

#include "Sample/Sample.hpp"
#include "SegmentPlane/SegmentPlane.hpp"

namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t = double>
class LinearTestSampleGenerator {
  std::size_t GenCount = 0, ErrCount = 0;
  std::mt19937_64 Engine;
  std::uniform_real_distribution<svm_float_t> FloatDistribution;
  SegmentPlane<Dimension, svm_float_t> Segmentation;
  svm_float_t FlipPossibility, FlipDistance;

  std::function<svm_float_t()> get_float = [&] {
    return FloatDistribution(Engine);
  };

 public:
  explicit LinearTestSampleGenerator(std::size_t = 0, svm_float_t = 0,
                                     svm_float_t = 0, svm_float_t = -1,
                                     svm_float_t = 1);
  Sample<Dimension, svm_float_t> operator()();

  SegmentPlane<Dimension, svm_float_t> &GetSegmentation() {
    return Segmentation;
  }
  svm_float_t FaultRate() {
    return svm_float_t(GenCount - ErrCount) / GenCount;
  }
};

}  // namespace SVM

//////////implementation//////////

namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t>
LinearTestSampleGenerator<Dimension, svm_float_t>::LinearTestSampleGenerator(
    std::size_t seed, svm_float_t _FlipPossibility, svm_float_t _FlipDistance,
    svm_float_t distribution_lowerbound, svm_float_t distribution_upperbound)
    : Engine(seed),
      FloatDistribution(distribution_lowerbound, distribution_upperbound),
      FlipPossibility(_FlipPossibility),
      FlipDistance(_FlipDistance) {
  std::ranges::generate(Segmentation.weight, get_float);
  Segmentation.bias = get_float() / 4;
};

template <std::size_t Dimension, std::floating_point svm_float_t>
Sample<Dimension, svm_float_t>
LinearTestSampleGenerator<Dimension, svm_float_t>::operator()() {
  GenCount++;
  Sample<Dimension, svm_float_t> sample;
  std::ranges::generate(sample.data, get_float);
  auto &&classfication = Segmentation.weight * sample.data + Segmentation.bias;

  static std::uniform_real_distribution<svm_float_t> RandDistribuion(0, 1);
  if (RandDistribuion(Engine) < FlipPossibility &&
      std::abs(classfication) <
          FlipDistance * std::sqrt(Segmentation.weight * Segmentation.weight)) {
    classfication *= -1;
    ErrCount++;
  }

  sample.classification = sgn(classfication);
  return sample;
};

}  // namespace SVM

#endif