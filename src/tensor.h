#pragma once
#include <fmt/format.h>
#include <time.h>

#include <initializer_list>
#include <numeric>
#include <string>
#include <vector>

#include "curand.h"

namespace detail {
template <typename T, size_t show_elements = 3>
std::string to_string(const std::vector<T>& vec) {
  if (vec.empty()) {
    return fmt::format("[] (size: 0)");
  }

  if (vec.size() <= 2 * show_elements) {
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) {
        result += ", ";
      }
      result += fmt::format("{}", vec[i]);
    }
    result += fmt::format("] (size: {})", vec.size());
    return result;
  }

  std::string result = "[";

  // 显示前show_elements个元素
  for (size_t i = 0; i < show_elements; ++i) {
    result += fmt::format("{}, ", vec[i]);
  }

  // 显示省略号
  result += "... ";

  // 显示后show_elements个元素
  for (size_t i = vec.size() - show_elements; i < vec.size(); ++i) {
    if (i > vec.size() - show_elements) {
      result += ", ";
    }
    result += fmt::format("{}", vec[i]);
  }

  result += fmt::format("] (size: {})", vec.size());
  return result;
  // return fmt::format("[{}]", fmt::join(vec, ", "));
}
}  // namespace detail

template <typename T>
class GPUTensor;

class Tensor {
 public:
  Tensor(std::initializer_list<int> shape, size_t size)
      : shape_(shape), size_(size), data_(nullptr) {}

  Tensor(const std::vector<int>& shape, size_t size)
      : shape_(shape), size_(size), data_(nullptr) {}

  virtual ~Tensor() = default;

  template <typename T>
  GPUTensor<T>& cast() {
    return static_cast<GPUTensor<T>&>(*this);
  }

  template <typename T>
  const GPUTensor<T>& cast() const {
    return static_cast<const GPUTensor<T>&>(*this);
  }

  Tensor(Tensor&& rhs) {
    this->shape_ = std::move(rhs.shape_);
    this->size_ = rhs.size_;
    this->data_ = rhs.data_;

    rhs.data_ = nullptr;
    rhs.size_ = 0;
  }

 protected:
  std::vector<int> shape_;
  size_t size_;
  void* data_;
};

template <typename T>
class GPUTensor : public Tensor {
 public:
  GPUTensor(std::initializer_list<int> shape)
      : Tensor(shape, std::accumulate(shape.begin(), shape.end(), 1,
                                      std::multiplies<int>{})) {
    cudaMalloc(&data_, size_ * sizeof(T));
  }

  GPUTensor(const std::vector<int>& shape)
      : Tensor(shape, std::accumulate(shape.begin(), shape.end(), 1,
                                      std::multiplies<int>{})) {
    cudaMalloc(&data_, size_ * sizeof(T));
  }

  ~GPUTensor() {
    if (data_) cudaFree(data_);
  }

  GPUTensor(GPUTensor&& rhs) : Tensor(std::move(rhs)) {}

  void random_uniform(T low = 0, T high = 1,
                      unsigned long long seed = time(nullptr)) {
    static_assert(
        std::is_same<T, float>::value || std::is_same<T, double>::value,
        "random_uniform only supports float/double tensors");
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    if constexpr (std::is_same<T, float>::value) {
      curandGenerateUniform(gen, (float*)data_, size_);
    } else if constexpr (std::is_same<T, double>::value) {
      curandGenerateUniformDouble(gen, (double*)data_, size_);
    }
    curandDestroyGenerator(gen);
  }

  std::vector<T> to_host() const {
    std::vector<T> res;
    res.resize(size_);
    cudaMemcpy(res.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
    return res;
  }

  T* mutable_data() { return reinterpret_cast<T*>(data_); }
  const T* data() const { return reinterpret_cast<const T*>(data_); }

  std::string to_string() const {
    const int show_elements = 3;
    auto vec = to_host();
    if (vec.empty()) {
      return fmt::format("[] (size: 0)");
    }

    if (vec.size() <= 2 * show_elements) {
      std::string result = "[";
      for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) {
          result += ", ";
        }
        result += fmt::format("{}", vec[i]);
      }
      result += fmt::format("] (size: {})", vec.size());
      return result;
    }

    std::string result = "[";

    // 显示前show_elements个元素
    for (size_t i = 0; i < show_elements; ++i) {
      result += fmt::format("{}, ", vec[i]);
    }

    // 显示省略号
    result += "... ";

    // 显示后show_elements个元素
    for (size_t i = vec.size() - show_elements; i < vec.size(); ++i) {
      if (i > vec.size() - show_elements) {
        result += ", ";
      }
      result += fmt::format("{}", vec[i]);
    }

    result += fmt::format("] (size: {})", vec.size());
    return result;
    // return fmt::format("[{}]", fmt::join(vec, ", "));
  }

  const std::vector<int>& shape() const { return shape_; }
  size_t size() const { return size_; }
};
