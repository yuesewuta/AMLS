#pragma once
// 统一日志封装（spdlog）。
//
// 用法（fmt 格式串，标签手写进格式串）：
//   AMLS_LOG_INFO("[ge] 层 {} 完成，稠密块 {:.2f}GB", l, gb);
//   AMLS_LOG_WARN("[cond] p={} 超过稠密上限", p);
//   AMLS_LOG_DEBUG("[ge] ...");   // 低于 info 级时不求值开销极小
//
// 默认 sink = stdout（彩色），pattern = "时间 级别 线程 消息"。
// 统计 CSV 等文件输出另行自建 basic_file_sink logger，不走全局 logger。
#include <memory>
#include <string>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace aams {
namespace log {

inline std::shared_ptr<spdlog::logger>& global() {
    static std::shared_ptr<spdlog::logger> logger = [] {
        auto lg = spdlog::stdout_color_mt("amls");
        lg->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");
        lg->set_level(spdlog::level::info);
        return lg;
    }();
    return logger;
}

/// 命令行控制：debug|info|warn|error。
inline void setLevel(const std::string& lvl) {
    global()->set_level(spdlog::level::from_str(lvl));
}

} // namespace log
} // namespace aams

#define AMLS_LOG_INFO(...) aams::log::global()->info(__VA_ARGS__)
#define AMLS_LOG_WARN(...) aams::log::global()->warn(__VA_ARGS__)
#define AMLS_LOG_ERROR(...) aams::log::global()->error(__VA_ARGS__)
#define AMLS_LOG_DEBUG(...) aams::log::global()->debug(__VA_ARGS__)
