#ifndef __SVM_SMO_HPP__
#define __SVM_SMO_HPP__
#include <algorithm>
#include <cstddef>
#include <deque>
#include <random>
#include <utility>

#include "../SVM/SVM.hpp"
#include "../common/common.hpp"

namespace SVM {

template <std::size_t DataSetSize, std::size_t Dimension,
          std::floating_point svm_float_t>
void SMO(SVM<DataSetSize, Dimension, svm_float_t>& svm, svm_float_t Tolerance,
         std::size_t EpochLimit, std::size_t seed,
         const DataCallback<std::size_t>& EpochCallback,
         const DataCallback<decltype(svm_float_t())>& ModifyCallback) {
  std::mt19937 Engine(seed);
  std::uniform_real_distribution<svm_float_t> RealDistribution(-1, 1);
  std::ranges::generate(svm.lambda, [&] { return RealDistribution(Engine); });

  FixedVector<DataSetSize, svm_float_t> E;
  for (int i = 0; i < DataSetSize; i++)
    E[i] = svm(svm.sample[i].data) - svm.sample[i].classification;

  std::deque<int> violate;
  int index_when_no_violate = 0;
  auto is_violate = [&](int i, svm_float_t e) {
    if (sgn(e) == -1 || sgn(Tolerance - e) == 1) return 2;
    auto& yi = svm.sample[i].classification;
    auto g_yi = (E[i] + yi) * yi;
    if (sgn(e) == 0) return int(sgn(g_yi - 1) >= 0);
    if (sgn(Tolerance - e) == 0) return int(sgn(g_yi - 1) <= 0);
    return 0;
  };

  for (int i = 0; i < DataSetSize; i++) {
    auto&& v = is_violate(i, E[i]);
    if (v == 2)
      violate.push_front(i);
    else if (v == 1)
      violate.push_back(i);
  }

  for (std::size_t epoch = 0; epoch != EpochLimit; epoch++) {
    int i, j;
    if (!violate.empty()) {
      i = violate.front();
      violate.pop_front();
    } else {
      i = index_when_no_violate;
      index_when_no_violate = (index_when_no_violate + 1) % DataSetSize;
    }
    std::swap(E[0], E[i]);
    if (sgn(E[i]) > 0)
      j = std::min_element(E.begin() + 1, E.end()) - E.begin();
    else
      j = std::max_element(E.begin() + 1, E.end()) - E.begin();
    if (j == i) j = 0;
    std::swap(E[0], E[i]);

    auto &L_i = svm.lambda[i], &L_j = svm.lambda[j];
    const auto& [y_i, y_i_d] = svm.sample[i];
    const auto& [y_j, y_j_d] = svm.sample[j];

    auto L_i_unclip =
        L_i + y_i * (E[j] - E[i]) /
                  (svm.kernel(y_i_d, y_i_d) + svm.kernel(y_j_d, y_j_d) -
                   2 * svm.kernel(y_i_d, y_j_d));
    auto L_i_low = y_i == y_j ? std::max(svm_float_t(0), L_i + L_j - Tolerance)
                              : std::max(svm_float_t(0), L_i - L_j);
    auto L_i_high = y_i == y_j ? std::min(Tolerance, L_i + L_j)
                               : std::min(Tolerance, Tolerance + L_i - L_j);
    auto L_i_new = std::clamp(L_i_unclip, L_i_low, L_i_high);

    auto L_y_sum = L_i * y_i + L_j * y_j;
    auto L_j_new = (L_y_sum - L_i_new * y_i) * y_j;

    for (int t = 0; t < DataSetSize; t++) {
      E[t] += svm.sample[i].classification * (L_i_new - L_i) *
                  (svm.kernel(svm.sample[i].data, svm.sample[t].data)) +
              svm.sample[j].classification * (L_j_new - L_j) *
                  (svm.kernel(svm.sample[j].data, svm.sample[t].data));
    }

    auto modify = std::abs(L_i_new - L_i) + std::abs(L_j_new - L_j);

    L_i = L_i_new;
    L_j = L_j_new;

    for (int t : {i, j}) {
      auto&& v = is_violate(t, E[t]);
      if (v == 2)
        violate.push_front(t);
      else if (v == 1)
        violate.push_back(t);
    }

    EpochCallback(epoch);
    ModifyCallback(modify);
  }
  int supporting_vector_count = 0;
  svm.bias = 0;
  for (int t = 0; t < DataSetSize; t++) {
    if (sgn(svm.lambda[t]) == 0) continue;
    supporting_vector_count++;
    svm.bias -= E[t];
  }
  svm.bias /= supporting_vector_count;
}

}  // namespace SVM

#endif