// blas.hpp 中线程局部/全局状态的定义（见头内注释）。
#include "linalg/blas.hpp"

namespace amls {
namespace blas {

thread_local bool tls_starved{false};
std::atomic_flag g_mtBudgetLock = ATOMIC_FLAG_INIT;

} // namespace blas
} // namespace amls
