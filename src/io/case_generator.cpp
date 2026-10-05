#include "io/case_generator.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Sparse>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace amls {

PencilPair pathGraph(int64_t n, bool consistentMass) {
    PencilPair p;
    std::vector<Eigen::Triplet<double, int64_t>> kT, mT;
    kT.reserve(static_cast<size_t>(3 * n));
    mT.reserve(static_cast<size_t>(3 * n));
    for (int64_t i = 0; i < n; ++i) {
        kT.emplace_back(i, i, 2.0);
        if (consistentMass)
            mT.emplace_back(i, i, 2.0 / 3.0);
        else
            mT.emplace_back(i, i, 1.0);
        if (i + 1 < n) {
            kT.emplace_back(i, i + 1, -1.0);
            kT.emplace_back(i + 1, i, -1.0);
            if (consistentMass) {
                mT.emplace_back(i, i + 1, 1.0 / 6.0);
                mT.emplace_back(i + 1, i, 1.0 / 6.0);
            }
        }
    }
    p.K.resize(n, n);
    p.K.setFromTriplets(kT.begin(), kT.end());
    p.M.resize(n, n);
    p.M.setFromTriplets(mT.begin(), mT.end());
    return p;
}

DefaultTraits::Vector pathGraphSpectrum(int64_t n, bool consistentMass) {
    DefaultTraits::Vector w(n);
    for (int64_t j = 1; j <= n; ++j) {
        const double th = M_PI * static_cast<double>(j) /
                          static_cast<double>(n + 1);
        double lambda = 2.0 - 2.0 * std::cos(th);
        if (consistentMass)
            lambda = 6.0 * (1.0 - std::cos(th)) / (2.0 + std::cos(th));
        w[j - 1] = lambda;
    }
    std::sort(w.data(), w.data() + n);
    return w;
}

PencilPair gridGraph2d(int64_t nx, int64_t ny) {
    const int64_t n = nx * ny;
    PencilPair p;
    std::vector<Eigen::Triplet<double, int64_t>> kT;
    kT.reserve(static_cast<size_t>(5 * n));
    auto id = [nx](int64_t i, int64_t j) { return j * nx + i; };
    for (int64_t j = 0; j < ny; ++j) {
        for (int64_t i = 0; i < nx; ++i) {
            const int64_t v = id(i, j);
            kT.emplace_back(v, v, 4.0);
            if (i > 0)     kT.emplace_back(v, id(i - 1, j), -1.0);
            if (i + 1 < nx) kT.emplace_back(v, id(i + 1, j), -1.0);
            if (j > 0)     kT.emplace_back(v, id(i, j - 1), -1.0);
            if (j + 1 < ny) kT.emplace_back(v, id(i, j + 1), -1.0);
        }
    }
    p.K.resize(n, n);
    p.K.setFromTriplets(kT.begin(), kT.end());
    p.M.resize(n, n);
    p.M.setIdentity();
    return p;
}

DefaultTraits::Vector gridGraph2dSpectrum(int64_t nx, int64_t ny) {
    DefaultTraits::Vector w(nx * ny);
    int64_t c = 0;
    for (int64_t j = 1; j <= ny; ++j) {
        const double ty = 2.0 - 2.0 * std::cos(M_PI * static_cast<double>(j) /
                                               static_cast<double>(ny + 1));
        for (int64_t i = 1; i <= nx; ++i) {
            const double tx = 2.0 - 2.0 * std::cos(M_PI * static_cast<double>(i) /
                                                   static_cast<double>(nx + 1));
            w[c++] = tx + ty;
        }
    }
    std::sort(w.data(), w.data() + nx * ny);
    return w;
}

PencilPair gridGraph3d(int64_t nx, int64_t ny, int64_t nz) {
    const int64_t n = nx * ny * nz;
    PencilPair p;
    std::vector<Eigen::Triplet<double, int64_t>> kT;
    kT.reserve(static_cast<size_t>(7 * n));
    auto id = [nx, ny](int64_t i, int64_t j, int64_t k) {
        return (k * ny + j) * nx + i;
    };
    for (int64_t k = 0; k < nz; ++k) {
        for (int64_t j = 0; j < ny; ++j) {
            for (int64_t i = 0; i < nx; ++i) {
                const int64_t v = id(i, j, k);
                kT.emplace_back(v, v, 6.0);
                if (i > 0)      kT.emplace_back(v, id(i - 1, j, k), -1.0);
                if (i + 1 < nx) kT.emplace_back(v, id(i + 1, j, k), -1.0);
                if (j > 0)      kT.emplace_back(v, id(i, j - 1, k), -1.0);
                if (j + 1 < ny) kT.emplace_back(v, id(i, j + 1, k), -1.0);
                if (k > 0)      kT.emplace_back(v, id(i, j, k - 1), -1.0);
                if (k + 1 < nz) kT.emplace_back(v, id(i, j, k + 1), -1.0);
            }
        }
    }
    p.K.resize(n, n);
    p.K.setFromTriplets(kT.begin(), kT.end());
    p.M.resize(n, n);
    p.M.setIdentity();
    return p;
}

DefaultTraits::Vector gridGraph3dSpectrum(int64_t nx, int64_t ny,
                                          int64_t nz) {
    auto axis = [](int64_t m, double* out) {
        for (int64_t j = 1; j <= m; ++j)
            out[j - 1] = 2.0 - 2.0 * std::cos(M_PI * static_cast<double>(j) /
                                              static_cast<double>(m + 1));
    };
    std::vector<double> wx(static_cast<size_t>(nx)), wy(static_cast<size_t>(ny)),
        wz(static_cast<size_t>(nz));
    axis(nx, wx.data());
    axis(ny, wy.data());
    axis(nz, wz.data());
    DefaultTraits::Vector w(nx * ny * nz);
    int64_t c = 0;
    for (int64_t k = 0; k < nz; ++k)
        for (int64_t j = 0; j < ny; ++j)
            for (int64_t i = 0; i < nx; ++i)
                w[c++] = wx[static_cast<size_t>(i)] +
                         wy[static_cast<size_t>(j)] +
                         wz[static_cast<size_t>(k)];
    std::sort(w.data(), w.data() + nx * ny * nz);
    return w;
}

} // namespace amls
