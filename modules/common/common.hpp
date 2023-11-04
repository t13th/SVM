#ifndef __SVM_COMMON_HPP__
#define __SVM_COMMON_HPP__

#ifdef __USE_EIGEN__
#include <Eigen/Eigen>

#include "Eigen/src/Core/Matrix.h"
#else
#include <algorithm>
#include <array>
#include <initializer_list>
#include <numeric>
#endif
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>

namespace SVM {

// 对Eigen封装或直接实现向量类
template <std::size_t Dimension, std::floating_point svm_float_t = double>
class FixedVector {
  using value_type = svm_float_t;
#ifdef __USE_EIGEN__
  Eigen::Vector<svm_float_t, (int)Dimension> content;
  using iterator = decltype(content.begin());
  using const_iterator = decltype(content.cbegin());
#else
  using iterator = value_type*;
  using const_iterator = const value_type*;
  std::array<svm_float_t, Dimension> content;
#endif
 public:
  FixedVector(const std::initializer_list<value_type>& v) {
    std::ranges::copy(v, content.begin());
  }
  FixedVector() = default;
  FixedVector(const FixedVector&) = default;

  FixedVector operator+(const FixedVector<Dimension, svm_float_t>&) const;
  FixedVector operator-(const FixedVector<Dimension, svm_float_t>&) const;
  svm_float_t dot(const FixedVector<Dimension, svm_float_t>&) const;
  FixedVector operator*(const svm_float_t&) const;

  constexpr const_iterator begin() const { return content.begin(); }
  constexpr const_iterator end() const { return content.end(); }
  constexpr iterator begin() { return content.begin(); }
  constexpr iterator end() { return content.end(); }
  svm_float_t& operator[](int index) { return content[index]; }
};

using ClassificationType = int;
const double ClassificationEps = 1e-6;

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
}  // namespace SVM

#ifndef __USE_EIGEN__
template <std::size_t Dimension, std::floating_point svm_float_t>
svm_float_t FixedVector<Dimension, svm_float_t>::dot(
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
#else
template <std::size_t Dimension, std::floating_point svm_float_t>
svm_float_t FixedVector<Dimension, svm_float_t>::dot(
    const FixedVector<Dimension, svm_float_t>& a) const {
  return content.dot(a.content);
}
template <std::size_t Dimension, std::floating_point svm_float_t>
FixedVector<Dimension, svm_float_t>
FixedVector<Dimension, svm_float_t>::operator+(
    const FixedVector<Dimension, svm_float_t>& b) const {
  auto a = *this;
  a.content += b.content;
  return a;
}
template <std::size_t Dimension, std::floating_point svm_float_t>
FixedVector<Dimension, svm_float_t>
FixedVector<Dimension, svm_float_t>::operator-(
    const FixedVector<Dimension, svm_float_t>& b) const {
  auto a = *this;
  a.content -= b.content;
  return a;
}
template <std::size_t Dimension, std::floating_point svm_float_t>
FixedVector<Dimension, svm_float_t>
FixedVector<Dimension, svm_float_t>::operator*(const svm_float_t& k) const {
  auto a = *this;
  a.content *= k;
  return a;
}
#endif
};  // namespace SVM

#endif