#include <iostream>
#include <vector>
#include <stdlib.h>

#include <atomic>
#include <thread>
#include <mutex>

#include "Timer.h"
#include "TaskQueue.h"
 
std::atomic<uint64_t> count(0);
std::atomic<bool> all_task_enqueued(false);
std::unique_ptr<TaskQueue> task_queue;
int n_schedule_levels = 0;

void solve(int n_queens, int col, std::vector<int> &board)
{
	if (col == n_queens) {
        ++count;
		return;
	}
 
	for (int i = 0, j = 0; i < n_queens; i++) {
		for (j = 0; j < col; j++) {
            if (board[j] == i || abs(board[j] - i) == col - j) // the new queue i attacks j
                break;
        }
		if (j < col) continue;
 
		board[col] = i;
		solve(n_queens, col + 1, board);
	}
}

void schedule(int n_queens, int col, std::vector<int> &board)
{
	for (int i = 0, j = 0; i < n_queens; i++) {
		for (j = 0; j < col; j++) {
            if (board[j] == i || abs(board[j] - i) == col - j) // the new queue i attacks j
                break;
        }
		if (j < col) continue;
 
		board[col] = i;
        task_queue->push(Task{n_queens, col + 1, board});
	}
}

void process()
{
    Task task;
    while (!all_task_enqueued) {
        while (task_queue->pop(task)) {
            if (task.col < n_schedule_levels) {
                schedule(task.n_queens, task.col, task.board);
            }
            else {
                all_task_enqueued = true;
                solve(task.n_queens, task.col, task.board);
            }
        }
    }
}
 
int main(int argc, char **argv)
{
    int n_queens = 0;
    int n_threads;
    if (argc <= 3 ||
        (n_queens = atoi(argv[1])) <= 0 ||
        (n_threads         = atoi(argv[2])) <= 0 ||
        (n_schedule_levels = atoi(argv[3])) <= 0  )
    {
        std::cerr << "Usage: " << argv[0] << " <n_queens> <n_threads> <n_schedule_levels>" << std::endl;
        return 1;
    }

    size_t max_q_len = 1;
    for (int i = 0; i < n_schedule_levels; ++i)
        max_q_len *= n_queens;
    task_queue = std::unique_ptr<TaskQueue>(new TaskQueue(max_q_len));

    std::vector<std::thread> thread_pool;
    for (int i = 0; i < n_threads; ++i) {
        thread_pool.emplace_back(process);
    }

    Timer timer;
    timer.start_timer();
    task_queue->push(Task{n_queens, 0, std::vector<int>(n_queens)});
    for (int i = 0; i < n_threads; ++i)
        thread_pool[i].join();
    timer.end_timer();

    std::cout << "Num of solutions: " << count << std::endl;
    std::cout << "Num of total tasks = " << task_queue->n_total_tasks() << std::endl;
    std::cout << "Time = " << timer.get_timed_interval() << " us" << std::endl;
}
