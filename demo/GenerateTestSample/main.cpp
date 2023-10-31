#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../../modules/LinearTestSampleGenerator/LinearTestSampleGenerator.hpp"
#include "../../modules/Sample/Sample.hpp"

const int Dimension = 5;
const int n = 30;

void output(double x) {
  std::cout << std::fixed << std::setprecision(6) << std::setw(9) << x << " ";
}
int main() {
  std::vector<SVM::Sample<Dimension>> data;
  SVM::LinearTestSampleGenerator<Dimension> gen(
      std::chrono::system_clock::now().time_since_epoch().count());
  for (int i = 0; i < n; i++) data.push_back(gen());
  auto Seg = gen.GetSegmentation();
  std::ranges::for_each(Seg.weight, output);
  std::cout << "+ ";
  output(Seg.bias);
  std::cout << std::endl;
  for (auto& v : data) {
    std::cout << std::setw(2) << v.classification - 1 << ":( ";
    std::ranges::for_each(v.data, output);
    std::cout << ")" << std::endl;
  }
}