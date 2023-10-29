#ifndef __SVM_SMO_HPP__
#define __SVM_SMO_HPP__
#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

#include "../SVM/SVM.hpp"
#include "../common/common.hpp"
namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t,
          std::random_access_iterator ForwardIterator>
void SMO(
    SVM<Dimension, svm_float_t>&, const ForwardIterator&,
    const ForwardIterator&, svm_float_t, svm_float_t, size_t,
    std::vector<svm_float_t>& = std::vector<svm_float_t>(), size_t = 0,
    const std::function<void(size_t)>& = [](size_t) {},
    const std::function<void(double)>& = [](double) {});

}  // namespace SVM

//////////Implementation//////////

namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t,
          std::random_access_iterator ForwardIterator>
void SMO(SVM<Dimension, svm_float_t>& svm, const ForwardIterator& first,
         const ForwardIterator& last, svm_float_t Tolerance,
         svm_float_t DifferenceLimit, size_t EpochLimit,
         std::vector<svm_float_t>& Lambda, size_t seed,
         const std::function<void(size_t)>& ProgressCallback,
         const std::function<void(double)>& DifferenceCallback) {
  const auto& DataSet = first;

  auto& Weight = svm.segmentation.weight;
  auto& Bias = svm.segmentation.bias;

  int n = std::distance(first, last);

  std::mt19937 engine(seed);
  std::uniform_real_distribution<svm_float_t> LambdaDistribution(-1.0, 1.0);
  if (Lambda.empty()) {
    Lambda.resize(n);
    std::ranges::generate(Lambda, [&]() { return LambdaDistribution(engine); });
  }
  /*
  std::transform(std::execution::seq,first, last, Lambda.begin(), [=](const
  auto& a) { if (a.classification == ClassificationType::Positive) return
  (svm_float_t)negative_cnt / n; else return (svm_float_t)positive_cnt / n;
  });
   */

  size_t Epoch = 0;
  while (Epoch < EpochLimit) {
    ProgressCallback(Epoch++);

    auto sum = std::transform_reduce(
        first, last, Lambda.begin(), FixedVector<Dimension, svm_float_t>(),
        std::plus<>{}, [](const auto& v, const auto& L) {
          return v.data * (L * (static_cast<int>(v.classification) - 1));
        });
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        if (i == j) continue;
        auto y_i = static_cast<int>(DataSet[i].classification) - 1;
        const auto& x_i = DataSet[i].data;
        auto& L_i = Lambda[i];
        auto y_j = static_cast<int>(DataSet[j].classification) - 1;
        const auto& x_j = DataSet[j].data;
        auto& L_j = Lambda[j];

        auto tsum = sum - x_i * L_i * y_i - x_j * L_j * y_j;
        auto L_j_low = y_i == y_j
                           ? std::max(svm_float_t(0), L_i + L_j - Tolerance)
                           : std::max(svm_float_t(0), L_j - L_i);
        auto L_j_high = y_i == y_j ? std::min(Tolerance, L_i + L_j)
                                   : std::min(Tolerance, Tolerance + L_j - L_i);
        auto L_j_new =
            std::clamp(L_j + y_j * (sum * (x_i - x_j) - y_i + y_j) /
                                 (x_i * x_i + x_j * x_j - x_i * x_j * 2),
                       L_j_low, L_j_high);
        L_i += y_i * y_j * (L_j - L_j_new);
        L_j = L_j_new;
        sum = tsum + x_i * (L_i * y_i) + x_j * (L_j * y_j);
      }
    auto weight_difference = Weight;
    Weight = sum;
    int supporting_vector_cnt;
    Bias = 0;
    auto max_bias = DBL_MIN, min_bias = DBL_MAX;
    std::for_each(first, last, [&](const auto& v) {
      auto&& val = v.data * Weight;
      if (v.classification == Negative && val > max_bias) max_bias = val;
      if (v.classification == Positive && val < min_bias) min_bias = val;
    });
    Bias = -(max_bias + min_bias) / 2;
    weight_difference = weight_difference - Weight;
    auto difference = weight_difference * weight_difference;
    DifferenceCallback(difference);
    if (difference < DifferenceLimit * DifferenceLimit) break;
  }
}

}  // namespace SVM

#endif