#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <mutex>
#include <ostream>
#include <thread>
#include <vector>

#include "../../modules/Optimizer/SMO.hpp"
#include "../../modules/SVM/SVM.hpp"
#include "../../modules/Sample/Sample.hpp"
#include "../../modules/TestSampleGenerator/TestSampleGenerator.hpp"

const int Dimension = 2;
const int n = 2000;

void output(double x) {
  std::cout << std::setprecision(6) << std::setw(9) << x << " ";
}

int main() {
  size_t seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::vector<SVM::Sample<Dimension>> data;
  SVM::TestSampleGenerator<Dimension> gen(seed);
  for (int i = 0; i < n; i++) data.push_back(gen());
  auto Seg = gen.GetSegmentation();

  std::ranges::for_each(Seg.weight, output);
  std::cout << "+ ";
  output(Seg.bias);
  std::cout << std::endl;
  for (int cnt = 0; auto& v : data) {
    std::cout << std::setw(2) << v.classification - 1 << ":( ";
    std::ranges::for_each(v.data, output);
    std::cout << ")" << std::endl;
    if (cnt++ > 10) {
      std::cout << "..." << std::endl;
      break;
    }
  }

  SVM::SVM<Dimension> svm;

  size_t progress = 0;
  double difference = 0;
  const size_t EpochLimit = 50;

  std::mutex mtx;
  std::condition_variable cv;
  bool SMOFinished = false;

  auto SMOFunc = [&]() {
    SVM::SMO(
        svm, data.begin(), data.end(), 1e1, 1e-13, EpochLimit, seed,
        [&](size_t _progress) { progress = _progress; },
        [&](double _difference) { difference = _difference; });
    std::unique_lock<std::mutex> lock(mtx);
    SMOFinished = true;
    cv.notify_one();
  };

  size_t bar_len = 80;
  std::jthread smo_thread(SMOFunc);
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::cout.put('\r');
    int curser_pos = 0;
    std::cout << std::setprecision(5) << std::setw(5)
              << double(int(double(progress) / EpochLimit * 10000)) / 100
              << "% |";
    for (; curser_pos * EpochLimit < progress * bar_len; curser_pos++)
      std::cout.put('*');
    for (; curser_pos < bar_len; curser_pos++) std::cout.put(' ');
    std::cout << "| " << difference << "            " << std::flush;
    {
      std::unique_lock<std::mutex> lock(mtx);
      if (SMOFinished) {
        std::cout << std::endl;
        break;
      }
    }
  }

  int correct_cnt = std::ranges::count_if(data, [&](const auto& sample) {
    return svm(sample.data) == sample.classification;
  });
  std::cout << "Classify accuracy:" << std::setprecision(5) << std::setw(5)
            << double(int(double(correct_cnt) / n * 10000)) / 100 << "%"
            << std::endl;
  std::ranges::for_each(svm.segmentation.weight, output);
  std::cout << "+ ";
  output(svm.segmentation.bias);

  std::fstream out("result.csv", std::ios::out);
  out << svm.segmentation.weight[0] << "," << svm.segmentation.weight[1] << ","
      << svm.segmentation.bias << std::endl;
  for (auto [c, p] : data)
    out << p[0] << "," << p[1] << "," << static_cast<int>(c) - 1 << std::endl;
  out.close();

  return 0;
}