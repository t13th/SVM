#ifndef __LINEAR_SVM_SMO_HPP__
#define __LINEAR_SVM_SMO_HPP__
#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <random>

#include "SVM/SVM.hpp"
#include "Sample/Sample.hpp"
#include "common/common.hpp"
namespace SVM {

// 基于向量运算和数值乘法可以交换顺序的性质进行优化
template <std::size_t DataSetSize, std::size_t Dimension,
          std::floating_point svm_float_t>
void LinearSMO(SVM<DataSetSize, Dimension, svm_float_t>& svm,
               svm_float_t Tolerance, std::size_t EpochLimit,
               svm_float_t ModifyLimit, std::size_t seed,
               const DataCallback<std::size_t>& EpochCallback,
               const DataCallback<decltype(svm_float_t())>& ModifyCallback) {
  using vector_t = FixedVector<Dimension, svm_float_t>;

  std::mt19937 engine(seed);
  std::uniform_real_distribution<svm_float_t> LambdaDistribution(-1.0, 1.0);

  std::ranges::generate(svm.lambda,
                        [&]() { return LambdaDistribution(engine); });
  svm.lambda[0] += -std::transform_reduce(
      svm.sample.begin(), svm.sample.end(), svm.lambda.begin(), svm_float_t(0),
      std::plus<>{},
      [](const Sample<Dimension, svm_float_t>& v, const svm_float_t& L) {
        return L * v.classification;
      });

  // 合并所有x_i到sum
  vector_t sum = std::transform_reduce(
      svm.sample.begin(), svm.sample.end(), svm.lambda.begin(), vector_t(),
      std::plus<>{},
      [](const Sample<Dimension, svm_float_t>& v, const svm_float_t& L) {
        return v.data * (L * v.classification);
      });

  for (std::size_t epoch = 0; epoch != EpochLimit; epoch++) {
    svm_float_t modify = 0;
    for (int i = 0; i < DataSetSize; i++)
      for (int j = 0; j < DataSetSize; j++) {
        if (svm.sample[i].classification == svm.sample[j].classification)
          continue;
        svm_float_t& L_i = svm.lambda[i];
        svm_float_t& L_j = svm.lambda[j];
        const auto& [y_i, x_i] = svm.sample[i];
        const auto& [y_j, x_j] = svm.sample[j];

        vector_t tsum = sum - x_i * L_i * y_i - x_j * L_j * y_j;
        svm_float_t L_j_low =
            y_i == y_j ? std::max(svm_float_t(0), L_i + L_j - Tolerance)
                       : std::max(svm_float_t(0), L_j - L_i);
        svm_float_t L_j_high = y_i == y_j
                                   ? std::min(Tolerance, L_i + L_j)
                                   : std::min(Tolerance, Tolerance + L_j - L_i);
        svm_float_t L_j_new = std::clamp(
            L_j + y_j * (sum.dot(x_i - x_j) - y_i + y_j) /
                      (x_i.dot(x_i) + x_j.dot(x_j) - x_i.dot(x_j) * 2),
            L_j_low, L_j_high);

        svm_float_t L_y_sum = L_i * y_i + L_j * y_j;
        svm_float_t L_i_new = (L_y_sum - L_j_new * y_j) * y_i;
        modify += std::abs(L_i_new - L_i) + std::abs(L_j_new - L_j);

        L_i = L_i_new;
        L_j = L_j_new;
        // 差分更新
        sum = tsum + x_i * (L_i * y_i) + x_j * (L_j * y_j);
      }
    // 报告回调
    ModifyCallback(modify);
    EpochCallback(epoch);
    if (modify < ModifyLimit) break;
  }
  // 比较bias范围
  svm_float_t min_bias_positive = std::numeric_limits<svm_float_t>::max(),
              max_bias_positive = -std::numeric_limits<svm_float_t>::max();
  svm_float_t min_bias_negative = std::numeric_limits<svm_float_t>::max(),
              max_bias_negative = -std::numeric_limits<svm_float_t>::max();
  for (int t = 0; t < DataSetSize; t++) {
    if (sgn(svm.lambda[t]) == 0) continue;
    svm_float_t v = svm.sample[t].data.dot(sum);
    if (svm.sample[t].classification == 1) {
      if (v > max_bias_positive) max_bias_positive = v;
      if (v < min_bias_positive) min_bias_positive = v;
    } else {
      if (v > max_bias_negative) max_bias_negative = v;
      if (v < min_bias_negative) min_bias_negative = v;
    }
  }
  if (std::abs(max_bias_positive - min_bias_negative) >
      std::abs(min_bias_positive - max_bias_negative))
    svm.bias = -(min_bias_positive + max_bias_negative) / 2;
  else
    svm.bias = -(min_bias_negative + max_bias_positive) / 2;
}

}  // namespace SVM

#endif