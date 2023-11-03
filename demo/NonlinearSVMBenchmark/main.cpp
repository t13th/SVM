#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <iostream>
#include <mutex>
#include <ostream>
#include <thread>

#include "SVM.hpp"

const int Dimension = 2;
const int n = 500;

void output(double x) {
  std::cout << std::setprecision(6) << std::setw(9) << x << " ";
}

int main() {
  std::cout << std::fixed;

  std::size_t seed =
      std::chrono::system_clock::now().time_since_epoch().count();
  std::vector<SVM::Sample<Dimension>> data;
  SVM::MoonTestSampleGenerator<> gen(seed, 0.75);
  for (int i = 0; i < n; i++) data.push_back(gen());

  for (int cnt = 0; auto& v : data) {
    std::cout << std::setw(2) << v.classification << ":( ";
    std::for_each_n(v.data.begin(), std::min(10, (int)Dimension), output);
    std::cout << ")" << std::endl;
    if (cnt++ > 10) {
      std::cout << "..." << std::endl;
      break;
    }
  }

  SVM::SVM<n, Dimension> svm(data.begin(), [](const auto& a, const auto& b) {
    auto&& d = a - b;
    return std::exp(d * d / 0.25 * (-1));
  });

  std::size_t progress = 0;
  double difference = 0;
  const std::size_t EpochLimit = 20;

  std::mutex mtx;
  std::condition_variable cv;
  bool SMOFinished = false;

  auto SMOFunc = [&]() {
    SVM::SMO(
        svm, 1e0, EpochLimit, 1.8e-13, seed,
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

  int correct_cnt = std::ranges::count_if(data, [&](const auto& sample) {
    return svm(sample.data) == sample.classification;
  });
  std::cout << "Classify accuracy:" << std::setprecision(2) << std::setw(6)
            << double(correct_cnt) / n * 100 << "%" << std::endl;

  std::fstream out("result.csv", std::ios::out);
  for (int i = 0; auto [c, p] : data)
    out << p[0] << "," << p[1] << "," << c << ","
        << (SVM::sgn(svm.lambda[i++]) ? 3 : 1) << std::endl;
  const int Slice = 1000;
  for (int i = 0; i < Slice; i++)
    for (int j = 0; j < Slice; j++) {
      double x = i, y = j;
      x = -4.5 + 9 * (x / Slice);
      y = -4.5 + 9 * (y / Slice);
      SVM::FixedVector<2> f;
      f.content = {x, y};
      out << svm(f) << std::endl;
    }
  out.close();

  return 0;
}