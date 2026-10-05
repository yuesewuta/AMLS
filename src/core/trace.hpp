#pragma once
// 轻量任务级打桩器：Chrome Trace JSON 输出（ui.perfetto.dev 直接打开）。
// 设计目标：零第三方依赖、记录期无锁（每线程事件缓冲）、未启用时开销
// 可忽略（一次原子读）。每条事件 = (类别, 任务名, 工作线程, 起止微秒,
// 树节点 id, 附加量×2)。附加量语义由埋点处定义（通常为祖先 id / 列数 /
// 树层级），供离线分析脚本聚合。
//
// 启用：环境变量 AMLS_TRACE=<输出.json>；进程退出前调 trace::dump()。
// 对照能力等价于 ctrack（同为 chrome-trace 输出），自维护以避免第三方
// 拉取/构建风险（Tracy 需服务端+GUI，MinGW 构建链过重，不采用）。
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace amls {
namespace trace {

struct Event {
    const char* cat;    // "ge" / "cond" / "solver"
    const char* name;   // "F","B","C","D","E","DONE","SPEC","PROJ.mid",...
    int64_t tid = -1;   // 线程登记序号（每个工作线程一条泳道）
    int64_t t0Us = 0;
    int64_t t1Us = 0;
    int64_t node = -1;  // 树节点 id（无节点语义 = -1）
    int64_t aux = -1;   // 埋点定义（如祖先 id / 保留列数）
    int64_t aux2 = -1;  // 埋点定义（通常为树层级）
};

namespace detail {

inline int64_t nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct Recorder {
    std::atomic<bool> enabled{false};
    std::string outPath;
    std::mutex regMutex;
    std::deque<std::vector<Event>*> buffers;   // 各线程缓冲的注册表
    std::atomic<int64_t> nDropped{0};

    static Recorder& inst() {
        static Recorder r;
        return r;
    }
    static std::vector<Event>& localBuf() {
        static thread_local std::vector<Event>* buf = nullptr;
        if (!buf) {
            buf = new std::vector<Event>();
            buf->reserve(1 << 16);
            Recorder& r = inst();
            std::lock_guard<std::mutex> lk(r.regMutex);
            r.buffers.push_back(buf);
        }
        return *buf;
    }
    static int64_t laneId() {
        static thread_local int64_t id = -1;
        if (id < 0) {
            static std::atomic<int64_t> next{0};
            id = next.fetch_add(1);
        }
        return id;
    }
};

} // namespace detail

inline bool enabled() { return detail::Recorder::inst().enabled.load(); }

/// 进程启动时调用一次（solver / 探针 main）：读环境变量决定启用与否。
inline void initFromEnv(const char* var = "AMLS_TRACE") {
    const char* p = std::getenv(var);
    if (p && *p) {
        detail::Recorder::inst().outPath = p;
        detail::Recorder::inst().enabled.store(true);
    }
}

/// 记录一条事件（t1==t0 表示瞬时标记）。埋点热路径直接用 Scope 更省事。
inline void add(const Event& e) {
    if (!enabled()) return;
    detail::Recorder::localBuf().push_back(e);
}

/// RAII 区间打点：构造记 t0，析构记 t1 并入库。可在区间内改写
/// scope.e.aux / aux2（先写好 node/cat/name）。
struct Scope {
    Event e;
    Scope(const char* cat, const char* name, int64_t node) {
        if (enabled()) {
            e.cat = cat;
            e.name = name;
            e.node = node;
            e.tid = detail::Recorder::laneId();
            e.t0Us = detail::nowUs();
        }
    }
    ~Scope() {
        if (enabled()) {
            e.t1Us = detail::nowUs();
            detail::Recorder::localBuf().push_back(std::move(e));
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

/// 瞬时标记（零时长事件）。
inline void mark(const char* cat, const char* name, int64_t node,
                 int64_t aux = -1, int64_t aux2 = -1) {
    if (!enabled()) return;
    Event e;
    e.cat = cat;
    e.name = name;
    e.node = node;
    e.aux = aux;
    e.aux2 = aux2;
    e.tid = detail::Recorder::laneId();
    e.t0Us = e.t1Us = detail::nowUs();
    detail::Recorder::localBuf().push_back(std::move(e));
}

/// 清空全部缓冲（多趟重跑之间调用，避免事件串趟）。
inline void reset() {
    if (!enabled()) return;
    detail::Recorder& r = detail::Recorder::inst();
    std::lock_guard<std::mutex> lk(r.regMutex);
    for (auto* b : r.buffers) b->clear();
}

/// 输出 Chrome Trace JSON（ph="X"）。未启用时空操作。
inline void dump(const char* pathOverride = nullptr) {
    detail::Recorder& r = detail::Recorder::inst();
    if (!r.enabled.load()) return;
    const std::string path =
        pathOverride ? pathOverride : r.outPath;
    if (path.empty()) return;
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    std::lock_guard<std::mutex> lk(r.regMutex);
    std::fprintf(f, "{\"traceEvents\":[\n");
    bool first = true;
    int64_t n = 0;
    for (const auto* b : r.buffers) {
        for (const Event& e : *b) {
            if (!first) std::fprintf(f, ",\n");
            first = false;
            // args 只携带非默认字段，控制文件体积
            std::fprintf(f,
                "{\"ph\":\"X\",\"cat\":\"%s\",\"name\":\"%s\",\"pid\":0,"
                "\"tid\":%lld,\"ts\":%lld,\"dur\":%lld",
                e.cat, e.name, (long long)e.tid, (long long)e.t0Us,
                (long long)(e.t1Us - e.t0Us));
            if (e.node >= 0 || e.aux >= 0 || e.aux2 >= 0) {
                std::fprintf(f, ",\"args\":{");
                bool fa = true;
                if (e.node >= 0) { std::fprintf(f, "\"node\":%lld", (long long)e.node); fa = false; }
                if (e.aux >= 0)  { std::fprintf(f, "%s\"aux\":%lld", fa ? "" : ",", (long long)e.aux); fa = false; }
                if (e.aux2 >= 0) { std::fprintf(f, "%s\"lvl\":%lld", fa ? "" : ",", (long long)e.aux2); }
                std::fprintf(f, "}");
            }
            std::fprintf(f, "}");
            ++n;
        }
    }
    std::fprintf(f, "\n]}\n");
    std::fclose(f);
    std::fprintf(stderr, "[trace] %lld events -> %s\n", (long long)n,
                 path.c_str());
}

} // namespace trace
} // namespace amls
