#ifndef __SVM_MOON_TEST_SAMPLE_GENERATOR_HPP__
#define __SVM_MOON_TEST_SAMPLE_GENERATOR_HPP__

#include <cstddef>
#include <cstdlib>
#include <random>

#include "Sample/Sample.hpp"

namespace SVM {

template <std::floating_point svm_float_t = double>
class MoonTestSampleGenerator {
  std::mt19937_64 Engine;
  std::uniform_real_distribution<svm_float_t> FloatDistribution{-2.5, 2.5};
  std::uniform_real_distribution<svm_float_t> SpreadDistribution;
  std::uniform_int_distribution<int> type{0, 1};

 public:
  explicit MoonTestSampleGenerator(std::size_t seed = 0,
                                   svm_float_t SpreadRange = 0)
      : Engine(seed), SpreadDistribution(-SpreadRange, SpreadRange){};
  Sample<2, svm_float_t> operator()();
};

}  // namespace SVM

//////////implementation//////////

namespace SVM {

template <std::floating_point svm_float_t>
Sample<2, svm_float_t> MoonTestSampleGenerator<svm_float_t>::operator()() {
  Sample<2, svm_float_t> sample;
  sample.classification = type(Engine) * 2 - 1;
  sample.data[0] = FloatDistribution(Engine);
  sample.data[1] = 2 * (sample.data[0] * sample.data[0] * 0.5 - 1.75) *
                   sample.classification;
  sample.data[0] += SpreadDistribution(Engine) + sample.classification;
  sample.data[1] += SpreadDistribution(Engine);
  return sample;
};

}  // namespace SVM

#endif