#include "pin_mem_allocator.h"

#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <mutex>

#include "cuda_runtime.h"

static inline uint64_t NextPowerOf2(size_t A) {
    A |= (A >> 1);
    A |= (A >> 2);
    A |= (A >> 4);
    A |= (A >> 8);
    A |= (A >> 16);
    A |= (A >> 32);
    return A + 1;
}

static inline uint64_t PowerOf2Ceil(size_t A) {
    if (!A) return 0;
    return NextPowerOf2(A - 1);
}

PinBlockCacheGroup::~PinBlockCacheGroup() {
    for (auto& kv : alloc_blocks_) {
        cudaFreeHost(kv.second.ptr);
    }

    for (auto& block : free_blocks_) {
        cudaFreeHost(block.ptr);
    }
}

PinBlockCacheGroup::PinBlockCacheGroup(size_t alloc_block_num,
                                       size_t per_block_size) {
    for (size_t i = 0; i < alloc_block_num; i++) {
        size_t alloc_size = PowerOf2Ceil(per_block_size);
        void* result;
        auto err = cudaHostAlloc(&result, alloc_size, cudaHostAllocDefault);
        if (err != cudaSuccess || result == nullptr) {
            continue;
        }
        free_blocks_.insert(Block{.size = alloc_size, .ptr = result});
    }
}

void* PinBlockCacheGroup::alloc(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    auto iter = free_blocks_.lower_bound(Block{.size = size, .ptr = nullptr});
    void* result{nullptr};
    if (iter == free_blocks_.end()) {
        // std::cerr << "not found valid size block: " << size << std::endl;
        size_t alloc_size = PowerOf2Ceil(size);
        auto err = cudaHostAlloc(&result, alloc_size, cudaHostAllocDefault);
        if (err != cudaSuccess || result == nullptr) {
            return nullptr;
        }
        alloc_blocks_.insert(
            {result, Block{.size = alloc_size, .ptr = result}});
        return result;
    }
    Block alloc_block{.size = iter->size, .ptr = iter->ptr};
    free_blocks_.erase(iter);
    alloc_blocks_.insert({alloc_block.ptr, alloc_block});
    return alloc_block.ptr;
}

void PinBlockCacheGroup::free(void* ptr) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto iter = alloc_blocks_.find(ptr);
    if (iter == alloc_blocks_.end()) {
        // std::cerr << "not found ptr: " << ptr << std::endl;
        return;
    }
    free_blocks_.insert(iter->second);
    alloc_blocks_.erase(iter);
}

void* PinBlockCache::alloc(size_t size) {
    auto idx = hash(size);
    auto cache_group = getCacheGroup(idx);
    auto* ptr = cache_group->alloc(size);
    // 保存ptr -> cache group idx
    setBlock2Group(ptr, idx);
    return ptr;
}

void PinBlockCache::free(void* ptr) {
    auto cache_group = getCacheGroup(ptr);
    cache_group->free(ptr);
    eraseBlock2Group(ptr);
}

