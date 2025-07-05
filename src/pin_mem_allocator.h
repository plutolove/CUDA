#pragma once
#include <folly/container/F14Map.h>

#include <cstdint>
#include <memory>
#include <set>

#include "folly/ConcurrentSkipList.h"
#include "folly/concurrency/ConcurrentHashMap.h"

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

class PinBlockCacheGroup {
 public:
  using BlockCache = std::set<Block, BlockComparator>;
  PinBlockCacheGroup(size_t alloc_block_num = 128);
  ~PinBlockCacheGroup();

  void* alloc(size_t size);

  void free(void* ptr);

 private:
  BlockCache free_blocks_{};
  std::unordered_map<void*, Block> alloc_blocks_{};
  mutable std::mutex mtx_;
};

class PinBlockCache {
 public:
  PinBlockCache(size_t group_num, size_t pre_alloc_block_per_group)
      : cache_group_size(group_num) {
    cache_groups_.resize(group_num);
    for (size_t i = 0; i < group_num; i++) {
      cache_groups_[i] =
          std::make_shared<PinBlockCacheGroup>(pre_alloc_block_per_group);
    }
  }

  void* alloc(size_t size);

  void free(void* ptr);

  size_t hash(size_t size) {
    static std::hash<size_t> hash{};
    return hash(size) % cache_group_size;
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
    // auto iter = block2group.find(ptr);
    // if (iter != block2group.end()) {
    //   std::cerr << "---- ptr repeated: " << ptr << "\t" << idx << std::endl;
    // }
    block2group.insert(ptr, idx);
  }
  void eraseBlock2Group(void* ptr) {
    // auto iter = block2group.find(ptr);
    // if (iter == block2group.end()) {
    //   std::cerr << "---- ptr not found: " << ptr << std::endl;
    // }
    block2group.erase(ptr);
  }

 private:
  size_t cache_group_size;
  std::vector<std::shared_ptr<PinBlockCacheGroup>> cache_groups_;
  folly::ConcurrentHashMap<void*, size_t> block2group{};
};

inline PinBlockCache& getThreadLocalCache() {
  static PinBlockCache cache_(32, 64);
  return cache_;
}

class PinMemAllocator {
 public:
  static void* alloc(size_t size) { return getThreadLocalCache().alloc(size); }

  static void free(void* ptr) { getThreadLocalCache().free(ptr); }
};
