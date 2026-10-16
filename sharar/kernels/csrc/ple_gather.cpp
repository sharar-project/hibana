// SPDX-License-Identifier: Apache-2.0
// PLE table rows from the host shards in one call, without the GIL, in two passes: preadv2(RWF_NOWAIT) copies every
// cached row (a syscall that never blocks), then the rows it left are read blocking, one per pool thread, so the
// misses' NVMe latencies overlap however few rows the call has. The
// Python gather paid ~2.4 us of interpreter work per row (mincore + memmove through ctypes) and serialized its
// thread pool's page faults on the GIL: ~50 ms per round for 8K rows at 128 requests.
#include <torch/extension.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>
#include <vector>

#ifndef RWF_NOWAIT
#define RWF_NOWAIT 0x00000008
#endif

namespace {

constexpr int64_t kChunk = 4;  // rows a worker claims at a time in the non-blocking pass
constexpr int64_t kRowsPerHelper = 16;  // rows per woken helper thread in the non-blocking pass

struct Job {
  const int32_t* fds;
  const int64_t* offsets;  // byte offset of each shard's first row in its file
  const int64_t* shard;
  const int64_t* local;
  uint8_t* out;
  int64_t n;
  int64_t row_bytes;
  int64_t* missed;  // rows the non-blocking pass left, read blocking in the second pass
  bool blocking = false;  // the second pass: next indexes `missed`
  std::atomic<int64_t> nmissed{0};
  std::atomic<int64_t> next{0};
  std::atomic<int64_t> hits{0};
  std::atomic<int64_t> misses{0};
  std::atomic<int64_t> errors{0};
};

std::atomic<bool> g_nowait{true};  // cleared when the filesystem refuses RWF_NOWAIT (ZFS: EOPNOTSUPP)

bool read_full(int fd, uint8_t* dst, int64_t size, int64_t off) {
  int64_t got = 0;
  while (got < size) {
    ssize_t r = pread(fd, dst + got, size - got, off + got);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return false;
    got += r;
  }
  return true;
}

void work(Job* job) {
  const int64_t rb = job->row_bytes;
  int64_t hits = 0, misses = 0, errors = 0;
  if (job->blocking) {  // one row at a time: each is an NVMe read
    const int64_t m = job->nmissed.load();
    for (int64_t k; (k = job->next.fetch_add(1)) < m;) {
      const int64_t i = job->missed[k], s = job->shard[i];
      if (read_full(job->fds[s], job->out + i * rb, rb, job->offsets[s] + job->local[i] * rb)) {
        ++misses;
      } else {
        ++errors;
      }
    }
  } else {
    for (;;) {
      int64_t lo = job->next.fetch_add(kChunk);
      if (lo >= job->n) break;
      int64_t hi = std::min(lo + kChunk, job->n);
      for (int64_t i = lo; i < hi; ++i) {
        const int64_t s = job->shard[i];
        struct iovec iov = {job->out + i * rb, static_cast<size_t>(rb)};
        ssize_t r = g_nowait.load(std::memory_order_relaxed)
                        ? preadv2(job->fds[s], &iov, 1, job->offsets[s] + job->local[i] * rb, RWF_NOWAIT) : -1;
        if (r == rb) {
          ++hits;
          continue;
        }
        if (r < 0 && errno == EOPNOTSUPP) g_nowait.store(false);
        // EAGAIN (not cached), a short read (a row across a page boundary, half cached): the blocking pass reads it
        job->missed[job->nmissed.fetch_add(1)] = i;
      }
    }
  }
  job->hits += hits;
  job->misses += misses;
  job->errors += errors;
}

class Pool {
 public:
  explicit Pool(int threads) {
    for (int t = 0; t < threads; ++t) workers_.emplace_back([this] { loop(); });
  }

  // Runs `job` on `helpers` workers (at most the pool's) and the calling thread; returns when all rows are read.
  // Only the helpers wake. One call at a time: concurrent callers wait their turn.
  void run(Job* job, int helpers) {
    std::lock_guard<std::mutex> call(call_mu_);
    helpers = std::min(helpers, static_cast<int>(workers_.size()));
    if (helpers > 0) {
      std::unique_lock<std::mutex> lock(mu_);
      job_ = job;
      tickets_ = helpers;
      pending_ = helpers;
    }
    for (int i = 0; i < helpers; ++i) cv_.notify_one();
    work(job);
    if (helpers > 0) {
      std::unique_lock<std::mutex> lock(mu_);
      done_.wait(lock, [this] { return pending_ == 0; });
      job_ = nullptr;
    }
  }

 private:
  void loop() {
    for (;;) {
      Job* job;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return tickets_ > 0; });
        --tickets_;
        job = job_;
      }
      work(job);
      std::unique_lock<std::mutex> lock(mu_);
      if (--pending_ == 0) done_.notify_one();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex call_mu_, mu_;
  std::condition_variable cv_, done_;
  Job* job_ = nullptr;
  int tickets_ = 0;
  int pending_ = 0;
};

Pool* pool(int threads) {
  static Pool* p = new Pool(threads);  // the first call's size; the threads live as long as the process
  return p;
}

// fds int32 [shards], offsets int64 [shards], shard / local int64 [n] (the row's shard and its row within it),
// out uint8 [n, row_bytes] contiguous. Returns (rows read without blocking, rows read blocking, rows that failed).
std::vector<int64_t> gather(torch::Tensor fds, torch::Tensor offsets, torch::Tensor shard, torch::Tensor local,
                            torch::Tensor out, int64_t threads) {
  TORCH_CHECK(fds.dtype() == torch::kInt32 && offsets.dtype() == torch::kInt64, "fds int32, offsets int64");
  TORCH_CHECK(shard.dtype() == torch::kInt64 && local.dtype() == torch::kInt64, "shard and local rows int64");
  TORCH_CHECK(out.dtype() == torch::kUInt8 && out.dim() == 2 && out.is_contiguous(), "out uint8 [n, row_bytes], contiguous");
  TORCH_CHECK(shard.numel() == out.size(0) && local.numel() == out.size(0), "one shard and local row per output row");
  TORCH_CHECK(fds.is_contiguous() && offsets.is_contiguous() && shard.is_contiguous() && local.is_contiguous(), "contiguous inputs");
  Job job;
  job.fds = fds.data_ptr<int32_t>();
  job.offsets = offsets.data_ptr<int64_t>();
  job.shard = shard.data_ptr<int64_t>();
  job.local = local.data_ptr<int64_t>();
  job.out = out.data_ptr<uint8_t>();
  job.n = out.size(0);
  job.row_bytes = out.size(1);
  std::vector<int64_t> missed(job.n);
  job.missed = missed.data();
  if (job.n > 0) {
    pybind11::gil_scoped_release release;
    Pool* p = pool(static_cast<int>(threads));
    // the non-blocking pass: one helper per kRowsPerHelper rows, so a few cached rows wake no thread
    p->run(&job, static_cast<int>(std::min<int64_t>((job.n + kRowsPerHelper - 1) / kRowsPerHelper - 1, threads)));
    const int64_t m = job.nmissed.load();
    if (m > 0) {  // the blocking pass: one thread per missed row, the caller included
      job.blocking = true;
      job.next = 0;
      p->run(&job, static_cast<int>(std::min<int64_t>(m - 1, threads)));
    }
  }
  return {job.hits.load(), job.misses.load(), job.errors.load()};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather", &gather, "PLE rows from the host shards: preadv2(RWF_NOWAIT), blocking pread for the misses, a native pool");
}
