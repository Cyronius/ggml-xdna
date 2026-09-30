// A fixed set of worker threads for the backend's host work. Starting threads
// on every call costs tens of microseconds each; the backend makes hundreds
// of parallel calls per prompt, so the threads are started once.
//
// The workers sleep on a condition variable between jobs rather than spin:
// spinning CPU threads slow the NPU (measured 30-50% under a busy CPU).
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

class thread_pool {
public:
    explicit thread_pool(int n_threads) {
        const int extra = std::max(0, n_threads - 1);  // the caller is the last worker
        for (int i = 0; i < extra; i++) workers_.emplace_back([this] { loop(); });
    }
    ~thread_pool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto & t : workers_) t.join();
    }
    int size() const { return (int) workers_.size() + 1; }

    // Runs fn(begin, end) over [0, n) in chunks, on every thread including
    // the caller's, and returns when all are done.
    void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)> & fn) {
        if (n <= 0) return;
        const int64_t parts = std::min<int64_t>(n, (int64_t) size() * 4);
        if (parts == 1 || workers_.empty()) { fn(0, n); return; }
        const int64_t chunk = (n + parts - 1) / parts;
        {
            // A worker that woke too late for the previous job can still be
            // inside work(), about to read chunk_ and n_. Changing them under
            // it would hand it the wrong range of this job: work skipped
            // (pending_ never reaches zero, and this waits forever) or done
            // twice. So wait until no worker is inside.
            std::unique_lock<std::mutex> lk(mu_);
            idle_cv_.wait(lk, [this] { return active_ == 0; });
            fn_ = &fn;
            n_ = n;
            chunk_ = chunk;
            next_ = 0;
            pending_ = (n + chunk - 1) / chunk;  // chunks that will actually run, which can be fewer than parts
            gen_++;
        }
        cv_.notify_all();
        work();
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this] { return pending_ == 0; });
    }

private:
    void work() {
        for (;;) {
            const int64_t b = next_.fetch_add(chunk_);
            if (b >= n_) return;
            (*fn_)(b, std::min(n_, b + chunk_));
            if (pending_.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> lk(mu_);
                done_cv_.notify_one();
            }
        }
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
                active_++;
            }
            work();
            std::lock_guard<std::mutex> lk(mu_);
            if (--active_ == 0) idle_cv_.notify_all();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_, idle_cv_;
    const std::function<void(int64_t, int64_t)> * fn_ = nullptr;
    int64_t n_ = 0, chunk_ = 1;
    std::atomic<int64_t> next_{ 0 }, pending_{ 0 };
    uint64_t gen_ = 0;
    int active_ = 0;  // workers inside work(), guarded by mu_
    bool stop_ = false;
};
