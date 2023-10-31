#ifndef __LINEAR_SVM_SMO_HPP__
#define __LINEAR_SVM_SMO_HPP__
#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <random>

#include "../SVM/SVM.hpp"
#include "../common/common.hpp"
namespace SVM {

template <std::size_t DataSetSize, std::size_t Dimension,
          std::floating_point svm_float_t>
void LinearSMO(SVM<DataSetSize, Dimension, svm_float_t>& svm,
               svm_float_t Tolerance, std::size_t EpochLimit,
               svm_float_t ModifyLimit, std::size_t seed,
               const DataCallback<std::size_t>& EpochCallback,
               const DataCallback<decltype(svm_float_t())>& ModifyCallback) {
  std::mt19937 engine(seed);
  std::uniform_real_distribution<svm_float_t> LambdaDistribution(-1.0, 1.0);
  std::ranges::generate(svm.lambda,
                        [&]() { return LambdaDistribution(engine); });
  svm.lambda[0] += -std::transform_reduce(
      svm.sample.begin(), svm.sample.end(), svm.lambda.begin(), svm_float_t(0),
      std::plus<>{},
      [](const auto& v, const auto& L) { return L * v.classification; });

  auto sum = std::transform_reduce(
      svm.sample.begin(), svm.sample.end(), svm.lambda.begin(),
      FixedVector<Dimension, svm_float_t>(), std::plus<>{},
      [](const auto& v, const auto& L) {
        return v.data * (L * v.classification);
      });

  for (std::size_t epoch = 0; epoch != EpochLimit; epoch++) {
    svm_float_t modify = 0;
    for (int i = 0; i < DataSetSize; i++)
      for (int j = 0; j < DataSetSize; j++) {
        if (svm.sample[i].classification == svm.sample[j].classification)
          continue;
        auto& L_i = svm.lambda[i];
        auto& L_j = svm.lambda[j];
        const auto& [y_i, x_i] = svm.sample[i];
        const auto& [y_j, x_j] = svm.sample[j];

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

        auto L_y_sum = L_i * y_i + L_j * y_j;
        auto L_i_new = (L_y_sum - L_j_new * y_j) * y_i;
        modify += std::abs(L_i_new - L_i) + std::abs(L_j_new - L_j);

        L_i = L_i_new;
        L_j = L_j_new;
        sum = tsum + x_i * (L_i * y_i) + x_j * (L_j * y_j);
      }
    ModifyCallback(modify);
    EpochCallback(epoch);
    if (modify < ModifyLimit) break;
  }
  svm_float_t min_bias = NAN, max_bias = NAN;
  for (int t = 0; t < DataSetSize; t++) {
    if (sgn(svm.lambda[t]) == 0) continue;
    auto v = svm.sample[t].data * sum;
    if ((svm.sample[t].classification == 1 && v > max_bias) ||
        std::isnan(max_bias))
      max_bias = v;
    else if ((svm.sample[t].classification == -1 && v < min_bias) ||
             std::isnan(min_bias))
      min_bias = v;
  }
  svm.bias = -(max_bias + min_bias) / 2;
}

}  // namespace SVM

#endif