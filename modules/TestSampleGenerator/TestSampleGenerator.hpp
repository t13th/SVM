#ifndef __TEST_SAMPLE_GENERATOR_HPP__
#define __TEST_SAMPLE_GENERATOR_HPP__

#include <algorithm>
#include <cstddef>
#include <functional>
#include <random>

#include "../Sample/Sample.hpp"
#include "../SegmentPlane/SegmentPlane.hpp"
#include "../common/common.hpp"

namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t = double>
class TestSampleGenerator {
  std::mt19937_64 Engine;
  std::uniform_real_distribution<svm_float_t> FloatDistribution;
  SegmentPlane<Dimension, svm_float_t> Segmentation;

  std::function<svm_float_t()> get_float = [&] {
    return FloatDistribution(Engine);
  };

 public:
  explicit TestSampleGenerator(size_t = 0, svm_float_t = -1, svm_float_t = 1);
  Sample<Dimension, svm_float_t> operator()();

  SegmentPlane<Dimension, svm_float_t> &GetSegmentation();
};

}  // namespace SVM

//////////implementation//////////

namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t>
TestSampleGenerator<Dimension, svm_float_t>::TestSampleGenerator(
    size_t seed, svm_float_t distribution_lowerbound,
    svm_float_t distribution_upperbound)
    : Engine(seed),
      FloatDistribution(distribution_lowerbound, distribution_upperbound) {
  std::ranges::generate(Segmentation.weight, get_float);
  Segmentation.bias = get_float() / 2;
};

template <std::size_t Dimension, std::floating_point svm_float_t>
Sample<Dimension, svm_float_t>
TestSampleGenerator<Dimension, svm_float_t>::operator()() {
  Sample<Dimension, svm_float_t> sample;
  std::ranges::generate(sample.data, get_float);
  auto &&classfication = Segmentation.weight * sample.data + Segmentation.bias;
  sample.classification =
      static_cast<ClassificationType>(sgn(classfication) + 1);
  return sample;
};

template <std::size_t Dimension, std::floating_point svm_float_t>
SegmentPlane<Dimension, svm_float_t> &
TestSampleGenerator<Dimension, svm_float_t>::GetSegmentation() {
  return Segmentation;
}

}  // namespace SVM

#endif