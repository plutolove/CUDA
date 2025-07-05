#pragma once
#include <time.h>

#include <boost/core/noncopyable.hpp>
#include <functional>
#include <numeric>
#include <ranges>
#include <string>
#include <vector>

#include "boost/noncopyable.hpp"
#include "cuda_runtime.h"
#include "curand.h"
#include "fmt/format.h"

template <typename T>
class GPUTensor;

class Tensor : boost::noncopyable {
 public:
  Tensor(const std::vector<int>& shape, size_t size)
      : shape_(shape), size_(size), data_(nullptr) {}

  template <typename T>
  GPUTensor<T>& cast() {
    return static_cast<GPUTensor<T>&>(*this);
  }

  template <typename T>
  const GPUTensor<T>& cast() const {
    return static_cast<const GPUTensor<T>&>(*this);
  }

 protected:
  std::vector<int> shape_;
  size_t size_;
  void* data_;
};

template <typename T>
class GPUTensor : public Tensor {
 public:
  GPUTensor(const std::vector<int>& shape)
      : Tensor(shape, std::accumulate(shape.begin(), shape.end(), 1,
                                      std::multiplies<int>{})) {
    cudaMalloc(&data_, size_ * sizeof(T));
  }

  ~GPUTensor() {
    if (data_) cudaFree(data_);
  }

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
    auto vec = to_host();
    return fmt::format("[{}]", fmt::join(vec, ", "));
  }

  const std::vector<int>& shape() const { return shape_; }
  size_t size() const { return size_; }
};
