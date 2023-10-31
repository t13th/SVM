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
               svm_float_t Tolerance, std::size_t EpochLimit, std::size_t seed,
               const DataCallback<std::size_t>& EpochCallback,
               const DataCallback<decltype(svm_float_t())>& ModifyCallback) {
  std::mt19937 engine(seed);
  std::uniform_real_distribution<svm_float_t> LambdaDistribution(-1.0, 1.0);
  std::ranges::generate(svm.lambda,
                        [&]() { return LambdaDistribution(engine); });

  auto sum = std::transform_reduce(
      svm.sample.begin(), svm.sample.end(), svm.lambda.begin(),
      FixedVector<Dimension, svm_float_t>(), std::plus<>{},
      [](const auto& v, const auto& L) {
        return v.data * (L * (static_cast<int>(v.classification) - 1));
      });

  size_t Epoch = 0;
  while (Epoch < EpochLimit) {
    EpochCallback(Epoch++);

    for (int i = 0; i < DataSetSize; i++)
      for (int j = 0; j < DataSetSize; j++) {
        if (i == j) continue;
        auto y_i = svm.sample[i].classification;
        const auto& x_i = svm.sample[i].data;
        auto& L_i = svm.lambda[i];
        auto y_j = svm.sample[j].classification;
        const auto& x_j = svm.sample[j].data;
        auto& L_j = svm.lambda[j];

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
        auto modify = std::abs(L_i_new - L_i) + std::abs(L_j_new - L_j);

        ModifyCallback(modify);

        L_i = L_i_new;
        L_j = L_j_new;
        sum = tsum + x_i * (L_i * y_i) + x_j * (L_j * y_j);
      }
  }
  int supporting_vector_count = 0;
  svm.bias = 0;
  for (int t = 0; t < DataSetSize; t++) {
    if (sgn(svm.lambda[t]) == 0) continue;
    supporting_vector_count++;
    svm.bias -= svm.sample[t].data * sum * svm.sample[t].classification;
  }
  svm.bias /= supporting_vector_count;
}

}  // namespace SVM

#endif