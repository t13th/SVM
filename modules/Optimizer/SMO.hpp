#ifndef __SVM_SMO_HPP__
#define __SVM_SMO_HPP__
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iostream>
#include <ostream>
#include <random>
#include <utility>

#include "../SVM/SVM.hpp"
#include "../common/common.hpp"

namespace SVM {

template <std::size_t DataSetSize, std::size_t Dimension,
          std::floating_point svm_float_t>
void SMO(SVM<DataSetSize, Dimension, svm_float_t>& svm, svm_float_t Tolerance,
         std::size_t EpochLimit, svm_float_t ModifyLimit, std::size_t seed,
         const DataCallback<std::size_t>& EpochCallback,
         const DataCallback<decltype(svm_float_t())>& ModifyCallback) {
  std::mt19937 Engine(seed);
  std::uniform_real_distribution<svm_float_t> RealDistribution(-1, 1);
  std::ranges::generate(svm.lambda, [&] { return RealDistribution(Engine); });
  svm.lambda[0] += -std::transform_reduce(
      svm.sample.begin(), svm.sample.end(), svm.lambda.begin(), svm_float_t(0),
      std::plus<>{},
      [](const auto& v, const auto& L) { return L * v.classification; });

  std::function<svm_float_t(int, int)> kernel;
  std::array<svm_float_t, DataSetSize * (1 + DataSetSize) / 2>* kernel_save =
      nullptr;
  if constexpr (DataSetSize * (1 + DataSetSize) / 2 * sizeof(svm_float_t) <=
                MaxMemUsage) {
    kernel_save =
        new std::array<svm_float_t, DataSetSize * (1 + DataSetSize) / 2>;
    for (int i = 0; i < DataSetSize; i++)
      for (int j = 0; j <= i; j++)
        (*kernel_save)[(i + 1) * i / 2 + j] =
            svm.kernel(svm.sample[i].data, svm.sample[j].data);
    kernel = [&](int i, int j) {
      if (i < j) std::swap(i, j);
      return (*kernel_save)[(i + 1) * i / 2 + j];
    };
  } else
    kernel = [&](int i, int j) {
      return svm.kernel(svm.sample[i].data, svm.sample[j].data);
    };

  FixedVector<DataSetSize, svm_float_t> E;
  for (int i = 0; i < DataSetSize; i++) {
    E[i] = svm.bias - svm.sample[i].classification;
    for (int j = 0; j < DataSetSize; j++)
      E[i] += svm.lambda[j] * svm.sample[j].classification * kernel(i, j);
  }

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

        auto L_j_low = y_i == y_j
                           ? std::max(svm_float_t(0), L_i + L_j - Tolerance)
                           : std::max(svm_float_t(0), L_j - L_i);
        auto L_j_high = y_i == y_j ? std::min(Tolerance, L_i + L_j)
                                   : std::min(Tolerance, Tolerance + L_j - L_i);
        auto L_j_new = std::clamp(
            L_j + y_j * (E[i] - E[j]) /
                      (kernel(i, i) + kernel(j, j) - 2 * kernel(i, j)),
            L_j_low, L_j_high);

        auto L_y_sum = L_i * y_i + L_j * y_j;
        auto L_i_new = (L_y_sum - L_j_new * y_j) * y_i;

        for (int t = 0; t < DataSetSize; t++) {
          E[t] += svm.sample[i].classification * (L_i_new - L_i) * kernel(i, t);
          E[t] += svm.sample[j].classification * (L_j_new - L_j) * kernel(j, t);
        }

        modify += std::abs(L_i_new - L_i) + std::abs(L_j_new - L_j);
        L_i = L_i_new;
        L_j = L_j_new;
      }
    ModifyCallback(modify);
    EpochCallback(epoch);
    if (modify < ModifyLimit) break;
  }
  svm_float_t min_bias = NAN, max_bias = NAN;
  for (int t = 0; t < DataSetSize; t++) {
    if (sgn(svm.lambda[t]) == 0) continue;
    auto v = E[t] + svm.sample[t].classification;
    if ((svm.sample[t].classification == 1 && v > max_bias) ||
        std::isnan(max_bias))
      max_bias = v;
    else if ((svm.sample[t].classification == -1 && v < min_bias) ||
             std::isnan(min_bias))
      min_bias = v;
  }
  svm.bias = -(max_bias + min_bias) / 2;

  delete kernel_save;
}

}  // namespace SVM

#endif