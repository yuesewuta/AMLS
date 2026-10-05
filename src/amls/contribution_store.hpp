#pragma once
// OOC 贡献流存储：消去节点 x 对祖先块 T 的增量 ΔT 以有符号稠密块
// 追加到 T 的专属文件（group-2「update data」的按目标文件方案，
// OOC 论文：written, read and deleted sequentially）。
//
// 峰值内存 = 单层工作集（祖先累加块不再驻留 RAM）；
// 磁盘流量 ≈ 块总量 × 2（写一次 + 消费时读一次）。
// 消费与追加从不并发（树层级 barrier 保证），同一目标的多线程
// 追加经条带锁串行化；open/close 每记录一次（记录为 MB 级大块，
// 系统调用开销可忽略）。
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_set>

#include "core/types.hpp"

namespace amls {

class ContributionStore {
public:
    explicit ContributionStore(std::string dir) : dir_(std::move(dir)) {
        std::filesystem::create_directories(dir_);
    }
    ~ContributionStore() { reset(); }

    ContributionStore(const ContributionStore&) = delete;
    ContributionStore& operator=(const ContributionStore&) = delete;

    /// 追加有符号贡献（线程安全；同 target 串行）。
    void append(int64_t target, const DefaultTraits::DenseMatrix& m) {
        const std::string p = path_(target);
        std::lock_guard<std::mutex> lk(stripes_[stripeOf_(target)]);
        std::ofstream os(p, std::ios::binary | std::ios::app);
        const int64_t r = m.rows(), c = m.cols();
        os.write(reinterpret_cast<const char*>(&r), sizeof(r));
        os.write(reinterpret_cast<const char*>(&c), sizeof(c));
        os.write(reinterpret_cast<const char*>(m.data()),
                 static_cast<std::streamsize>(r * c) *
                     static_cast<std::streamsize>(sizeof(double)));
        {
            std::lock_guard<std::mutex> lk2(setMutex_);
            targets_.insert(target);
        }
        bytes_ += static_cast<double>(r * c) * sizeof(double);
    }

    /// 读回 target 的全部贡献之和（按追加顺序）；无文件返回 false。
    /// consume=true 时删除文件（释放磁盘）。
    bool readSum(int64_t target, DefaultTraits::DenseMatrix& out,
                 bool consume) {
        const std::string p = path_(target);
        {
            std::lock_guard<std::mutex> lk(stripes_[stripeOf_(target)]);
            std::ifstream is(p, std::ios::binary);
            if (!is)
                return false;
            bool first = true;
            for (;;) {
                int64_t r, c;
                is.read(reinterpret_cast<char*>(&r), sizeof(r));
                if (!is)
                    break;
                is.read(reinterpret_cast<char*>(&c), sizeof(c));
                DefaultTraits::DenseMatrix m(r, c);
                is.read(reinterpret_cast<char*>(m.data()),
                        static_cast<std::streamsize>(r * c) *
                            static_cast<std::streamsize>(sizeof(double)));
                if (!is)
                    break;
                if (first) {
                    out = std::move(m);
                    first = false;
                } else {
                    out.noalias() += m;
                }
            }
            if (consume) {
                is.close();
                std::filesystem::remove(p);
                std::lock_guard<std::mutex> lk2(setMutex_);
                targets_.erase(target);
            }
        }
        return true;
    }

    void remove(int64_t target) {
        std::lock_guard<std::mutex> lk(stripes_[stripeOf_(target)]);
        std::filesystem::remove(path_(target));
        std::lock_guard<std::mutex> lk2(setMutex_);
        targets_.erase(target);
    }

    /// 删除全部贡献文件（每趟流水线重跑前调用）。
    void reset() {
        std::lock_guard<std::mutex> lk(setMutex_);
        for (int64_t t : targets_)
            std::filesystem::remove(path_(t));
        targets_.clear();
        bytes_ = 0.0;
    }

    double diskBytes() const { return bytes_; }

private:
    static size_t stripeOf_(int64_t k) {
        return (static_cast<size_t>(static_cast<uint64_t>(k) * 2654435761u) >>
                20) &
               255;
    }
    std::string path_(int64_t target) const {
        return dir_ + "/t" + std::to_string(target) + ".bin";
    }

    std::string dir_;
    std::array<std::mutex, 256> stripes_;
    std::mutex setMutex_;
    std::unordered_set<int64_t> targets_;
    double bytes_ = 0.0;
};

} // namespace amls
