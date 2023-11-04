#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "SVM.hpp"

void output(double x) {
  std::cout << std::setprecision(6) << std::setw(9) << x << " ";
}
int main() {
  std::cout << std::fixed;
  std::size_t seed =
      std::chrono::system_clock::now().time_since_epoch().count();
  // 读取不同的数据文件
  auto [train, test] = SVMDataLoader::BreastCancerWisconsin<32, 450, double>(
      std::string(PROJECT_ROOT) +
          "assets/datasets/breast+cancer+wisconsin+original/"
          // "breast-cancer-wisconsin.data",
          // "wpbc.data",
          "wdbc.data",
      seed);

  std::cout << std::endl;
  for (int cnt = 0; auto& v : train) {
    std::cout << std::setw(2) << v.classification << ":( ";
    std::ranges::for_each(v.data, output);
    std::cout << ")" << std::endl;
    if (cnt++ > 5) {
      std::cout << "..." << std::endl;
      break;
    }
  }
  std::cout << std::endl;
  for (int cnt = 0; auto& v : test) {
    std::cout << std::setw(2) << v.classification << ":( ";
    std::ranges::for_each(v.data, output);
    std::cout << ")" << std::endl;
    if (cnt++ > 5) {
      std::cout << "..." << std::endl;
      break;
    }
  }

  SVM::SVM<450, 30> svm(train.begin(), [](const SVM::FixedVector<30>& a,
                                          const SVM::FixedVector<30>& b) {
    SVM::FixedVector<30>&& d = a - b;
    return std::exp(d.dot(d) / 0.2 * (-1)) / 30;
    double x = a.dot(b);
    return x;
  });

  std::size_t progress = 0;
  double difference = 0;
  const std::size_t EpochLimit = 20;

  std::mutex mtx;
  std::condition_variable cv;
  bool SMOFinished = false;

  auto SMOFunc = [&]() {
    SVM::SMO(
        svm, 5e0, EpochLimit, 1e-100, seed,
        [&](std::size_t _progress) { progress = _progress; },
        [&](double _difference) { difference = _difference; });
    std::unique_lock<std::mutex> lock(mtx);
    SMOFinished = true;
    cv.notify_one();
  };

  std::size_t bar_len = 60;
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
  int correct_cnt = std::ranges::count_if(test, [&](const auto& sample) {
    return svm(sample.data) == sample.classification;
  });
  std::cout << "Classify accuracy:" << std::setprecision(3) << std::setw(6)
            << double(correct_cnt) / test.size() * 100 << "%" << std::endl;

  return 0;
}