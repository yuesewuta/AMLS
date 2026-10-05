#pragma once
// 极简线程池：fork-join 并行 for（std::thread，无 OpenMP 依赖）。
// 用于同层子结构 / 全体叶节点局部谱问题的并行（OOC 论文 §Parallel Computing）。
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

#include "linalg/blas.hpp"

namespace amls {

/// f(i) 对 [begin, end) 并行执行；numThreads<=0 时取硬件并发数。
template <class F>
void parallelFor(int64_t begin, int64_t end, F&& f, int numThreads = 0) {
    const int64_t n = end - begin;
    if (n <= 0)
        return;
    if (numThreads <= 0)
        numThreads = static_cast<int>(std::thread::hardware_concurrency());
    numThreads = std::max(1, std::min<int>(numThreads, static_cast<int>(n)));

    std::atomic<int64_t> next(begin);
    auto worker = [&]() {
        blas::pinThisThread();   // 本线程的 BLAS 并行度钉 1（nthreads-var 为
                                 // 每线程 ICV，主线程的设置传不到这里）
        for (;;) {
            const int64_t i = next.fetch_add(1);
            if (i >= end)
                break;
            f(i);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(numThreads - 1));
    for (int t = 0; t < numThreads - 1; ++t)
        pool.emplace_back(worker);
    worker();
    for (auto& th : pool)
        th.join();
}

} // namespace amls
