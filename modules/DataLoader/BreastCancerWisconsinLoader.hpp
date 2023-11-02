#ifndef __BREAST_CANCER_WISCONSIN_LOADER_HPP__
#define __BREAST_CANCER_WISCONSIN_LOADER_HPP__

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../Sample/Sample.hpp"

namespace SVMDataLoader {

template <std::size_t Dimension, std::size_t TrainDataSize,
          std::floating_point svm_float_t,
          class sample_t = SVM::Sample<Dimension - 2, svm_float_t>>
std::pair<std::vector<sample_t>, std::vector<sample_t>> BreastCancerWisconsin(
    const std::string& file_path, std::size_t seed = 0) {
  constexpr std::size_t DataSetSize = Dimension == 35   ? 198
                                      : Dimension == 32 ? 569
                                                        : 699;
  static_assert((Dimension == 35 || Dimension == 32 || Dimension == 11) &&
                TrainDataSize != 0 && TrainDataSize <= DataSetSize);

  std::ifstream data_file(file_path);

  if (!data_file.is_open())
    throw std::runtime_error("Fail to open data file at " + file_path + ".");

  auto get_sample = [&] {
    std::string line;
    std::getline(data_file, line);
    for (auto& c : line) c = c == ',' ? ' ' : c;
    std::istringstream is(line);
    auto get_number = [&] {
      std::string s;
      is >> s;
      try {
        return std::stoi(s);
      } catch (...) {
        return 0;
      }
    };
    sample_t sample;
    char classification;
    int ID;
    // 4,M,R
    ID = get_number();
    if constexpr (Dimension == 11) {
      for (auto& x : sample.data) x = get_number();
      is >> classification;
      sample.classification = classification == '4';
    } else {
      is >> classification;
      if constexpr (Dimension == 32)
        sample.classification = classification == 'M';
      if constexpr (Dimension == 35)
        sample.classification = classification == 'R';
      for (auto& x : sample.data) x = get_number();
    }
    if (sample.classification == 0) sample.classification = -1;
    return sample;
  };

  std::vector<sample_t> sample;
  sample.reserve(DataSetSize);
  for (int i = 0; i < DataSetSize; i++) sample.push_back(get_sample());

  static std::mt19937 Engine(seed);
  std::ranges::shuffle(sample, Engine);

  std::vector train_sample(sample.begin() + (DataSetSize - TrainDataSize),
                           sample.end());
  sample.resize(DataSetSize - TrainDataSize);
  sample.shrink_to_fit();
  data_file.close();
  return std::make_pair(train_sample, sample);
}

}  // namespace SVMDataLoader

#endif