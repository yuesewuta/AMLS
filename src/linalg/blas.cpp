// blas.hpp 中线程局部/全局状态的定义（见头内注释）。
#include "linalg/blas.hpp"

namespace amls {
namespace blas {

thread_local bool tls_starved{false};
thread_local bool tls_budgetActive{false};
thread_local std::function<int()> tls_availableThreads;
thread_local std::function<int(int)> tls_acquireThreads;
thread_local std::function<void(int)> tls_releaseThreads;
std::atomic_flag g_mtBudgetLock = ATOMIC_FLAG_INIT;

} // namespace blas
} // namespace amls
