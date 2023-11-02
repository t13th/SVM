#ifndef __SVM_COMMON_HPP__
#define __SVM_COMMON_HPP__

#ifdef __USE_EIGEN__
#include <Eigen/Eigen>
#else
#include <algorithm>
#include <numeric>
#endif
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>

namespace SVM {

template <std::size_t Dimension, std::floating_point svm_float_t = double>
struct FixedVector {
  using value_type = svm_float_t;

  FixedVector operator+(const FixedVector<Dimension, svm_float_t>&) const;
  FixedVector operator-(const FixedVector<Dimension, svm_float_t>&) const;
  svm_float_t operator*(const FixedVector<Dimension, svm_float_t>&) const;
  FixedVector operator*(const svm_float_t&) const;

#ifndef __USE_EIGEN__
  using iterator = value_type*;
  using const_iterator = const value_type*;
  std::array<svm_float_t, Dimension> content;
  constexpr const_iterator begin() const { return content.begin(); }
  constexpr const_iterator end() const { return content.end(); }
  constexpr iterator begin() { return content.begin(); }
  constexpr iterator end() { return content.end(); }
  svm_float_t& operator[](int index) { return content[index]; }
#else
  using iterator = value_type*;
  using const_iterator = const value_type*;
  Eigen::Vector<svm_float_t, (int)Dimension> content;
  constexpr auto begin() const { return content.begin(); }
  constexpr auto end() const { return content.end(); }
  constexpr auto begin() { return content.begin(); }
  constexpr auto end() { return content.end(); }
  svm_float_t& operator[](int index) { return content[index]; }

#endif
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

#ifndef __USE_EIGEN__
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
#else
template <std::size_t Dimension, std::floating_point svm_float_t>
svm_float_t FixedVector<Dimension, svm_float_t>::operator*(
    const FixedVector<Dimension, svm_float_t>& a) const {
  return a.content.dot(content);
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