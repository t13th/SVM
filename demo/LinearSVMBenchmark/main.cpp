#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <ostream>
#include <thread>
#include <vector>

#include "../../modules/LinearTestSampleGenerator/LinearTestSampleGenerator.hpp"
#include "../../modules/Optimizer/LinearSMO.hpp"
#include "../../modules/SVM/SVM.hpp"
#include "../../modules/Sample/Sample.hpp"

const int Dimension = 2;
const int n = 200;

void output(double x) {
  std::cout << std::setprecision(6) << std::setw(9) << x << " ";
}

int main() {
  std::cout << std::fixed;

  std::size_t seed =
      std::chrono::system_clock::now().time_since_epoch().count();
  std::vector<SVM::Sample<Dimension>> data;
  SVM::LinearTestSampleGenerator<Dimension, true> gen(seed);
  for (int i = 0; i < n; i++) data.push_back(gen(0.1, 5e-2));
  auto Seg = gen.GetSegmentation();

  std::ranges::for_each(Seg.weight, output);
  std::cout << "+ ";
  output(Seg.bias);
  std::cout << std::endl;
  for (int cnt = 0; auto& v : data) {
    std::cout << std::setw(2) << v.classification << ":( ";
    std::ranges::for_each(v.data, output);
    std::cout << ")" << std::endl;
    if (cnt++ > 10) {
      std::cout << "..." << std::endl;
      break;
    }
  }

  SVM::SVM<n, Dimension> svm(data.begin(), [](const auto& a, const auto& b) {
    auto&& d = a - b;
    return std::exp(d * d / 0.01 * (-1));
  });

  std::size_t progress = 0;
  double difference = 0;
  const std::size_t EpochLimit = 1e2;

  std::mutex mtx;
  std::condition_variable cv;
  bool SMOFinished = false;

  auto SMOFunc = [&]() {
    SVM::LinearSMO(
        svm, 1e0, EpochLimit, 3.35e-3, seed,
        [&](std::size_t _progress) { progress = _progress; },
        [&](double _difference) { difference = _difference; });
    std::unique_lock<std::mutex> lock(mtx);
    SMOFinished = true;
    cv.notify_one();
  };

  std::size_t bar_len = 80;
  std::jthread smo_thread(SMOFunc);
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::cout.put('\r');
    int curser_pos = 0;
    std::cout << std::setprecision(1) << std::setw(5)
              << double(progress) / EpochLimit * 100 << "% |";
    for (; curser_pos * EpochLimit < progress * bar_len; curser_pos++)
      std::cout.put('*');
    for (; curser_pos < bar_len; curser_pos++) std::cout.put(' ');
    std::cout << "| " << std::scientific << difference << std::fixed
              << "            " << std::flush;
    {
      std::unique_lock<std::mutex> lock(mtx);
      if (SMOFinished) {
        std::cout << std::endl;
        break;
      }
    }
  }

  SVM::LinearSVM<Dimension> lsvm(svm);

  int correct_cnt = std::ranges::count_if(data, [&](const auto& sample) {
    return lsvm(sample.data) == sample.classification;
  });
  std::cout << "Generator accuracy:" << std::setprecision(2) << std::setw(6)
            << gen.FaultRate() * 100 << "%" << std::endl;
  std::cout << "Classify accuracy:" << std::setprecision(2) << std::setw(6)
            << double(correct_cnt) / n * 100 << "%" << std::endl;
  std::ranges::for_each(lsvm.segmentation.weight, output);
  std::cout << "+ ";
  output(lsvm.segmentation.bias);
  std::cout << std::endl;

  std::fstream out("result.csv", std::ios::out);
  out << lsvm.segmentation.weight[0] << "," << lsvm.segmentation.weight[1]
      << "," << lsvm.segmentation.bias << std::endl;
  for (int i = 0; auto [c, p] : data)
    out << p[0] << "," << p[1] << "," << c << ","
        << (SVM::sgn(svm.lambda[i++]) ? 5 : 1) << std::endl;
  out.close();

  return 0;
}