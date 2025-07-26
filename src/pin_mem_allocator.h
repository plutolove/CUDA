#pragma once
#include <cstdint>
#include <set>

#include "folly/concurrency/ConcurrentHashMap.h"

class PinBlockCacheGroup {
 public:
  struct Block {
    size_t size;
    void* ptr{nullptr};
  };

  struct BlockComparator {
    bool operator()(const Block& lhs, const Block& rhs) const {
      if (lhs.size != rhs.size) {
        return lhs.size < rhs.size;
      }
      return reinterpret_cast<uintptr_t>(lhs.ptr) <
             reinterpret_cast<uintptr_t>(rhs.ptr);
    }
  };

  using BlockCache = std::set<Block, BlockComparator>;

  PinBlockCacheGroup(size_t alloc_block_num, size_t per_block_size);
  ~PinBlockCacheGroup();

  void* alloc(size_t size);

  void free(void* ptr);

  size_t getAllocBlockNum() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return alloc_blocks_.size();
  }

 private:
  BlockCache free_blocks_{};
  std::unordered_map<void*, Block> alloc_blocks_{};
  mutable std::mutex mtx_;
};

class PinBlockCache {
 public:
  PinBlockCache(size_t group_num, size_t pre_alloc_block_per_group,
                size_t per_block_size)
      : cache_group_size(group_num) {
    cache_groups_.resize(group_num);
    for (size_t i = 0; i < group_num; i++) {
      cache_groups_[i] = std::make_shared<PinBlockCacheGroup>(
          pre_alloc_block_per_group, per_block_size);
    }
  }

  void* alloc(size_t size);

  void free(void* ptr);

  // 防止size分布太集中，加上thread id
  // 适用于并发环境
  size_t hash(size_t size) {
    static std::hash<size_t> size_hash{};
    static std::hash<std::thread::id> tid_hash{};
    auto h1 = size_hash(size);
    auto h2 = tid_hash(std::this_thread::get_id());
    return ((h1 ^ (h2 << 1)) ^ (h2 >> 1)) % cache_group_size;
  }

  std::shared_ptr<PinBlockCacheGroup> getCacheGroup(size_t idx) {
    return cache_groups_[idx];
  }

  std::shared_ptr<PinBlockCacheGroup> getCacheGroup(void* ptr) {
    auto iter = block2group.find(ptr);
    if (iter == block2group.end()) {
      // std::cerr << "------ error not found group idx: " << ptr << std::endl;
      return nullptr;
    }
    return cache_groups_[iter->second];
  }

  void setBlock2Group(void* ptr, size_t idx) {
    auto iter = block2group.find(ptr);
    if (iter != block2group.end()) {
      // std::cerr << "---- ptr repeated: " << ptr << "\t" << idx << std::endl;
    }
    block2group.insert(ptr, idx);
  }

  void eraseBlock2Group(void* ptr) {
    auto iter = block2group.find(ptr);
    if (iter == block2group.end()) {
      // std::cerr << "---- ptr not found: " << ptr << std::endl;
    }
    block2group.erase(ptr);
  }

  size_t getAllocBlockNum() const {
    size_t result = 0;
    for (auto& group : cache_groups_) {
      result += group->getAllocBlockNum();
    }
    return result;
  }

 private:
  size_t cache_group_size;
  std::vector<std::shared_ptr<PinBlockCacheGroup>> cache_groups_;
  folly::ConcurrentHashMap<void*, size_t> block2group{};
};

inline PinBlockCache& getThreadLocalCache() {
  static PinBlockCache cache_(32, 64, 64);
  return cache_;
}

class PinMemAllocator {
 public:
  static void* alloc(size_t size) { return getThreadLocalCache().alloc(size); }

  static void free(void* ptr) { getThreadLocalCache().free(ptr); }

  static size_t getAllocBlockNum() {
    return getThreadLocalCache().getAllocBlockNum();
  }
};
