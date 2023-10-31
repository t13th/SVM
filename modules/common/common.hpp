#ifndef __SVM_COMMON_HPP__
#define __SVM_COMMON_HPP__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <numeric>

namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t = double>
struct FixedVector {
  using value_type = svm_float_t;
  using iterator = value_type*;
  using const_iterator = const value_type*;
  std::array<svm_float_t, Dimension> content;
  FixedVector operator+(const FixedVector<Dimension, svm_float_t>&) const;
  FixedVector operator-(const FixedVector<Dimension, svm_float_t>&) const;
  svm_float_t operator*(const FixedVector<Dimension, svm_float_t>&) const;
  FixedVector operator*(const svm_float_t&) const;

  constexpr const_iterator begin() const { return content.begin(); }
  constexpr const_iterator end() const { return content.end(); }
  constexpr iterator begin() { return content.begin(); }
  constexpr iterator end() { return content.end(); }
  std::array<svm_float_t, Dimension>* operator->() { return &content; }
  svm_float_t& operator[](int index) { return content[index]; }
};

using ClassificationType = int;
const double ClassificationEps = 1e-3;

template <std::floating_point svm_float_t = double>
int sgn(svm_float_t x, svm_float_t = ClassificationEps);

template <typename T>
using DataCallback = std::function<void(T)>;

const uint64_t MaxMemUsage = 1l << (10 + 10 + 10);

}  // namespace SVM

//////////Implementation//////////

namespace SVM {

template <std::floating_point svm_float_t>
int sgn(svm_float_t x, svm_float_t eps) {
  if (std::abs(x) > eps) return x > 0 ? 1 : -1;
  return 0;
}

template <std::size_t Dimension, std::floating_point svm_float_t>
svm_float_t FixedVector<Dimension, svm_float_t>::operator*(
    const FixedVector<Dimension, svm_float_t>& a) const {
  return std::transform_reduce(content.begin(), content.end(),
                               a.content.begin(), svm_float_t(0));
}
template <std::size_t Dimension, std::floating_point svm_float_t>
FixedVector<Dimension, svm_float_t>
FixedVector<Dimension, svm_float_t>::operator+(
    const FixedVector<Dimension, svm_float_t>& b) const {
  auto a = *this;
  std::transform(a.content.begin(), a.content.end(), b.content.begin(),
                 a.begin(), std::plus<>());
  return a;
}
template <std::size_t Dimension, std::floating_point svm_float_t>
FixedVector<Dimension, svm_float_t>
FixedVector<Dimension, svm_float_t>::operator-(
    const FixedVector<Dimension, svm_float_t>& b) const {
  auto a = *this;
  std::transform(a.content.begin(), a.content.end(), b.content.begin(),
                 a.begin(), std::minus<>());
  return a;
}
template <std::size_t Dimension, std::floating_point svm_float_t>
FixedVector<Dimension, svm_float_t>
FixedVector<Dimension, svm_float_t>::operator*(const svm_float_t& k) const {
  auto a = *this;
  std::ranges::transform(a, a.begin(), [&](const auto& x) { return x * k; });
  return a;
}

};  // namespace SVM

#endif