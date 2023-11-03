#ifndef __SVM_SVM_HPP__
#define __SVM_SVM_HPP__

#include <array>
#include <cstddef>
#include <functional>
#include <numeric>

#include "Sample/Sample.hpp"
#include "SegmentPlane/SegmentPlane.hpp"
#include "common/common.hpp"

namespace SVM {
//////////pre declaration//////////

template <std::size_t Dimension, std::floating_point svm_float_t = double>
struct LinearSVM;
template <std::size_t DataSetSize, std::size_t Dimension,
          std::floating_point svm_float_t = double>
class SVM;

template <std::size_t DataSetSize, std::size_t Dimension,
          std::floating_point svm_float_t = double>
void SMO(
    SVM<DataSetSize, Dimension, svm_float_t>&, svm_float_t, std::size_t,
    svm_float_t, std::size_t = 0,
    const DataCallback<std::size_t>& = [](std::size_t) {},
    const DataCallback<decltype(svm_float_t())>& = [](svm_float_t) {});

template <std::size_t DataSetSize, std::size_t Dimension,
          std::floating_point svm_float_t = double>
void LinearSMO(
    SVM<DataSetSize, Dimension, svm_float_t>&, svm_float_t, std::size_t,
    svm_float_t, std::size_t = 0,
    const DataCallback<std::size_t>& = [](std::size_t) {},
    const DataCallback<decltype(svm_float_t())>& = [](svm_float_t) {});

//////////end//////////

template <std::size_t DataSetSize, std::size_t Dimension,
          std::floating_point svm_float_t>
class SVM {
  using sample_t = Sample<Dimension, svm_float_t>;
  using data_t = decltype(sample_t().data);
  std::array<sample_t, DataSetSize> sample;
  svm_float_t bias;

  using kernel_function_t =
      std::function<svm_float_t(const data_t&, const data_t&)>;
  kernel_function_t kernel;

 public:
  friend struct LinearSVM<Dimension, svm_float_t>;
  friend void SMO<>(SVM<DataSetSize, Dimension, svm_float_t>&, svm_float_t,
                    std::size_t, svm_float_t, std::size_t,
                    const DataCallback<std::size_t>&,
                    const DataCallback<decltype(svm_float_t())>&);
  friend void LinearSMO<>(SVM<DataSetSize, Dimension, svm_float_t>&,
                          svm_float_t, std::size_t, svm_float_t, std::size_t,
                          const DataCallback<std::size_t>&,
                          const DataCallback<decltype(svm_float_t())>&);

 public:
  FixedVector<DataSetSize, svm_float_t> lambda;

  template <std::forward_iterator forwardIt>
  SVM(forwardIt, const kernel_function_t&);
  SVM() = delete;
  ClassificationType operator()(const FixedVector<Dimension, svm_float_t>&);
};

template <std::size_t Dimension, std::floating_point svm_float_t>
struct LinearSVM {
  SegmentPlane<Dimension, svm_float_t> segmentation;

  LinearSVM() = delete;
  template <std::size_t DataSetSize>
  LinearSVM(const SVM<DataSetSize, Dimension, svm_float_t>&);

  ClassificationType operator()(const FixedVector<Dimension, svm_float_t>&);
};

}  // namespace SVM

//////////Implementation//////////

namespace SVM {

template <std::size_t DataSetSize, std::size_t Dimension,
          std::floating_point svm_float_t>
template <std::forward_iterator forwardIt>
SVM<DataSetSize, Dimension, svm_float_t>::SVM(forwardIt first,
                                              const kernel_function_t& _kernel)
    : kernel(_kernel) {
  for (int i = 0; i < DataSetSize; i++) sample[i] = *first++;
}
template <std::size_t DataSetSize, std::size_t Dimension,
          std::floating_point svm_float_t>
ClassificationType SVM<DataSetSize, Dimension, svm_float_t>::operator()(
    const FixedVector<Dimension, svm_float_t>& data) {
  auto classfication = std::transform_reduce(
      sample.begin(), sample.end(), lambda.begin(), bias, std::plus<>{},
      [&](const sample_t& xi, const svm_float_t& l) {
        return xi.classification * l * kernel(xi.data, data);
      });
  return sgn(classfication);
}

template <std::size_t Dimension, std::floating_point svm_float_t>
template <std::size_t DataSetSize>
LinearSVM<Dimension, svm_float_t>::LinearSVM(
    const SVM<DataSetSize, Dimension, svm_float_t>& svm) {
  using sample_t = Sample<Dimension, svm_float_t>;
  segmentation.weight = std::transform_reduce(
      svm.sample.begin(), svm.sample.end(), svm.lambda.begin(),
      decltype(segmentation.weight)(), std::plus<>{},
      [&](const sample_t& xi, const svm_float_t& l) {
        return xi.data * l * xi.classification;
      });
  segmentation.bias = svm.bias;
}

template <std::size_t Dimension, std::floating_point svm_float_t>
ClassificationType LinearSVM<Dimension, svm_float_t>::operator()(
    const FixedVector<Dimension, svm_float_t>& data) {
  auto&& classfication = segmentation.weight * data + segmentation.bias;
  return sgn(classfication);
}

}  // namespace SVM
#endif