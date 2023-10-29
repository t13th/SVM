#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include "../../modules/Sample/Sample.hpp"
#include "../../modules/TestSampleGenerator/TestSampleGenerator.hpp"

const int Dimension = 5;
const int n = 30;

void output(double x) {
  std::cout << std::fixed << std::setprecision(6) << std::setw(9) << x << " ";
}
int main() {
  std::vector<SVM::Sample<Dimension>> data;
  SVM::TestSampleGenerator<Dimension> gen(
      std::chrono::system_clock::now().time_since_epoch().count());
  for (int i = 0; i < n; i++) data.push_back(gen());
  auto Seg = gen.GetSegmentation();
  std::ranges::for_each(Seg.get_weight(), output);
  std::cout << "+ ";
  output(Seg.get_bias());
  std::cout << std::endl;
  for (auto& v : data) {
    std::cout << std::setw(2) << v.get_type() * 2 - 1 << ":( ";
    std::ranges::for_each(v.get_data(), output);
    std::cout << ")" << std::endl;
  }
}