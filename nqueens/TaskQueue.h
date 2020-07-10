#pragma once

#include <cassert>
#include <mutex>
#include <condition_variable>

struct Task
{
    int n_queens;
    int col;
    std::vector<int> board;
};

class TaskQueue
{
public:
    TaskQueue(size_t max_size)
    : size_(max_size)
    , ring_buf_(max_size)
    , reader_idx_(0)
    , writer_idx_(0)
    , n_total_tasks_(0)
    { }

    void push(Task &&task)
    {
        std::unique_lock<std::mutex> lock(mtx_);

        size_t next = writer_idx_ + 1;
        if (next == size_)
            next = 0;

        if (next == reader_idx_)
            throw std::runtime_error("queue is full");

        std::swap(ring_buf_[writer_idx_], task);

        writer_idx_ = next;

        ++n_total_tasks_;
    }

    bool pop(Task &task)
    {
        std::unique_lock<std::mutex> lock(mtx_);

        if (reader_idx_ == writer_idx_)
            return false; // queue empty

        std::swap(task, ring_buf_[reader_idx_]);

        ++reader_idx_;
        if (reader_idx_ == size_)
            reader_idx_ = 0;

        return true;
    }

    size_t n_total_tasks() const { return n_total_tasks_; }

private:
    const size_t size_;

    std::vector<Task> ring_buf_;
    size_t reader_idx_; // position for the current read
    size_t writer_idx_; // position for the current write

    size_t n_total_tasks_;

    std::mutex mtx_;
};
