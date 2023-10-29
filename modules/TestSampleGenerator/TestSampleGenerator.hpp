#ifndef __TEST_SAMPLE_GENERATOR_HPP__
#define __TEST_SAMPLE_GENERATOR_HPP__

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>

#include "../Sample/Sample.hpp"
#include "../SegmentPlane/SegmentPlane.hpp"
#include "../common/common.hpp"

namespace SVM {

template <std::size_t Dimension, bool Flip = false,
          std::floating_point svm_float_t = double>
class TestSampleGenerator {
  size_t GenCount = 0, ErrCount = 0;
  std::mt19937_64 Engine;
  std::uniform_real_distribution<svm_float_t> FloatDistribution;
  SegmentPlane<Dimension, svm_float_t> Segmentation;

  std::function<svm_float_t()> get_float = [&] {
    return FloatDistribution(Engine);
  };

 public:
  explicit TestSampleGenerator(size_t = 0, svm_float_t = -1, svm_float_t = 1);
  Sample<Dimension, svm_float_t> operator()(svm_float_t = 0, svm_float_t = 0);

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

template <std::size_t Dimension, bool Flip, std::floating_point svm_float_t>
TestSampleGenerator<Dimension, Flip, svm_float_t>::TestSampleGenerator(
    std::size_t seed, svm_float_t distribution_lowerbound,
    svm_float_t distribution_upperbound)
    : Engine(seed),
      FloatDistribution(distribution_lowerbound, distribution_upperbound) {
  std::ranges::generate(Segmentation.weight, get_float);
  Segmentation.bias = get_float() / 4;
};

template <std::size_t Dimension, bool Flip, std::floating_point svm_float_t>
Sample<Dimension, svm_float_t>
TestSampleGenerator<Dimension, Flip, svm_float_t>::operator()(
    svm_float_t FlipPossibility, svm_float_t FlipDistance) {
  GenCount++;
  Sample<Dimension, svm_float_t> sample;
  std::ranges::generate(sample.data, get_float);
  auto &&classfication = Segmentation.weight * sample.data + Segmentation.bias;
  if constexpr (Flip) {
    static std::uniform_real_distribution<svm_float_t> RandDistribuion(0, 1);
    if (RandDistribuion(Engine) < FlipPossibility &&
        std::abs(classfication) <
            FlipDistance *
                std::sqrt(Segmentation.weight * Segmentation.weight)) {
      classfication *= -1;
      ErrCount++;
    }
  }
  sample.classification =
      static_cast<ClassificationType>(sgn(classfication) + 1);
  return sample;
};

}  // namespace SVM

#endif