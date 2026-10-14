// Standalone small-matrix benchmark; link exactly like an amls_lib consumer.
// Example (MSYS2 MinGW64): g++ -O3 -std=c++17 -Isrc -ID:/msys2/mingw64/include/eigen3
// tools/lapack_blas_bench.cpp build/libamls_lib.a D:/msys2/mingw64/lib/liblapack.a
// -lopenblas -lgfortran -lgomp -Wl,--wrap=dgemm_,--wrap=dsyrk_,--wrap=dsyr2k_,--wrap=dtrsm_,--wrap=dtrmm_,--wrap=dsymm_
// -Wl,--undefined=__wrap_dgemm_ -o build/lapack_blas_bench.exe
#include "core/dag.hpp"
#include "linalg/dense_eigensolver.hpp"

#include <chrono>
#include <cmath>
#include <future>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using DM = amls::DefaultTraits::DenseMatrix;
using Vec = amls::DefaultTraits::Vector;
using Clock = std::chrono::steady_clock;
using Solver = amls::LapackDenseEigensolver<amls::DefaultTraits>;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void serial(bool enabled) {
#ifdef _WIN32
    _putenv_s("AMLS_N7_SERIAL", enabled ? "1" : "");
#else
    if (enabled) setenv("AMLS_N7_SERIAL", "1", 1);
    else unsetenv("AMLS_N7_SERIAL");
#endif
}

struct Problem {
    DM k, m;
    Vec spectrum;
    explicit Problem(int n) : k(n, n), m(DM::Zero(n, n)), spectrum(n) {
        DM q(n, n);
        const double pi = std::acos(-1.0);
        for (int j = 0; j < n; ++j) {
            spectrum[j] = 1.0 + 2.0 * j / (n - 1);
            const double scale = std::sqrt((j == 0 ? 1.0 : 2.0) / n);
            for (int i = 0; i < n; ++i)
                q(i, j) = scale * std::cos(pi * (i + 0.5) * j / n);
            m(j, j) = 1.0 + static_cast<double>(j) / n;
        }
        k.noalias() = q * spectrum.asDiagonal() * q.transpose();
        Vec s = m.diagonal().array().sqrt();
        k = s.asDiagonal() * k * s.asDiagonal();
    }
    void check(const Vec& w, const DM& v, bool interval) const {
        const int first = interval ? static_cast<int>(spectrum.size()) / 3 : 0;
        const int count = interval ? static_cast<int>(spectrum.size()) / 3
                                   : static_cast<int>(spectrum.size());
        require(w.size() == count, "wrong eigenvalue count");
        const double eig = (w - spectrum.segment(first, count)).cwiseAbs().maxCoeff();
        DM residual = k * v - m * v * w.asDiagonal();
        const double rel = residual.norm() / (k.norm() * v.norm());
        DM gram = v.transpose() * m * v;
        const double orth = (gram - DM::Identity(count, count)).norm();
        std::cout << "correctness mode=" << (interval ? "interval" : "all")
                  << " count=" << count << " eig_error=" << eig
                  << " residual=" << rel << " M_orth=" << orth << '\n';
        require(eig < 2e-12 && rel < 2e-13 && orth < 2e-10,
                "generalized eigenpair correctness failed");
    }
    void solve(bool interval, Vec& w, DM& v) const {
        DM a = k, b = m;
        Solver solver;
        if (interval) {
            const int first = static_cast<int>(spectrum.size()) / 3;
            const int end = 2 * first;
            solver.solveInterval(a, b, (spectrum[first - 1] + spectrum[first]) / 2,
                                 (spectrum[end - 1] + spectrum[end]) / 2, w, v);
        } else solver.solveAll(a, b, w, v);
    }
};

void concurrentCheck(const Problem& p, bool forceSerial) {
    serial(forceSerial);
    const auto before = amls::blas::ScopedBudget::nBudgetGranted.load();
    amls::DagRunner dag(4);
    Vec wa, wb;
    DM va, vb;
    auto& all = dag.emplace([&] { p.solve(false, wa, va); });
    auto& part = dag.emplace([&] { p.solve(true, wb, vb); });
    dag.ready(all);
    dag.ready(part);
    dag.waitIdle();
    const auto granted = amls::blas::ScopedBudget::nBudgetGranted.load() - before;
    std::cout << "concurrent serial=" << forceSerial << " budgets=" << granted << '\n';
    require(forceSerial ? granted == 0 : granted > 0, "wrapper budget gating failed");
    p.check(wa, va, false);
    p.check(wb, vb, true);
}

void lateIdleCheck(const Problem& p) {
    serial(false);
    amls::DagRunner dag(4);
    std::mutex mutex;
    std::condition_variable changed;
    int arrived = 0;
    bool release = false;
    std::atomic<int> busyQueries{0}, idleQueries{0};
    const auto before = amls::blas::ScopedBudget::nBudgetGranted.load();
    Vec w;
    DM v;
    for (int i = 0; i < 3; ++i) {
        auto& task = dag.emplace([&, i] {
            {
                std::unique_lock<std::mutex> lock(mutex);
                ++arrived;
                changed.notify_all();
                changed.wait(lock, [&] { return arrived == 3; });
                if (i != 0) {
                    changed.wait(lock, [&] { return release; });
                    return;
                }
            }
            auto original = amls::blas::tls_acquireThreads;
            amls::blas::tls_acquireThreads = [&](int requested) {
                const int capacity = amls::blas::availableThreads();
                if (capacity == 1) ++busyQueries;
                else ++idleQueries;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    release = true;
                }
                changed.notify_all();
                return original(requested);
            };
            p.solve(false, w, v);
            amls::blas::tls_acquireThreads = std::move(original);
            require(amls::blas::omp_get_max_threads() == 1,
                    "wrapper did not restore worker ICV");
        });
        dag.ready(task);
    }
    dag.waitIdle();
    const auto granted = amls::blas::ScopedBudget::nBudgetGranted.load() - before;
    std::cout << "late_idle busy_queries=" << busyQueries << " idle_queries=" << idleQueries
              << " budgets=" << granted << '\n';
    require(busyQueries > 0 && idleQueries > 0 && granted > 0,
            "already-running LAPACK failed to acquire newly idle capacity");
    p.check(w, v, false);
}

void timing(const Problem& p, bool interval, int repeats) {
    double samples[2];
    for (int mode = 0; mode < 2; ++mode) {
        serial(mode == 0);
        amls::DagRunner dag(4);
        Vec w;
        DM v;
        auto solve = [&] {
            auto& task = dag.emplace([&] { p.solve(interval, w, v); });
            dag.ready(task);
            dag.waitIdle();
        };
        solve();
        const auto before = amls::blas::ScopedBudget::nBudgetGranted.load();
        const auto start = Clock::now();
        for (int r = 0; r < repeats; ++r) solve();
        samples[mode] = std::chrono::duration<double>(Clock::now() - start).count() / repeats;
        std::cout << "timing mode=" << (interval ? "interval" : "all")
                  << " serial=" << (mode == 0) << " seconds=" << samples[mode]
                  << " budgets=" << (amls::blas::ScopedBudget::nBudgetGranted.load() - before)
                  << '\n';
        p.check(w, v, interval);
    }
    std::cout << "speedup mode=" << (interval ? "interval" : "all")
              << " serial_over_MT=" << samples[0] / samples[1] << '\n';
}

int main(int argc, char** argv) {
    try {
        const int n = argc > 1 ? std::stoi(argv[1]) : 800;
        const int repeats = argc > 2 ? std::stoi(argv[2]) : 5;
        require(n >= 400 && n <= 1000 && repeats > 0, "dimension must be 400..1000");
        amls::blas::setNumThreads(1);
        std::cout << std::setprecision(9) << "OpenBLAS=" << openblas_get_config()
                  << " parallel=" << openblas_get_parallel() << " n=" << n << '\n';
        const Problem p(n);
        concurrentCheck(p, true);
        concurrentCheck(p, false);
        lateIdleCheck(p);
        timing(p, false, repeats);
        timing(p, true, repeats);
        serial(false);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "FAIL: " << ex.what() << '\n';
        serial(false);
        return 1;
    }
}
