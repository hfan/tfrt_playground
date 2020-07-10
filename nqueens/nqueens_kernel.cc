#include "tfrt/host_context/kernel_utils.h"

namespace tfrt {

int32_t solve(int n_queens, int col, std::vector<int> &board)
{
	if (col == n_queens) {
		return 1;
	}

    int32_t count = 0;
	for (int i = 0, j = 0; i < n_queens; i++) {
		for (j = 0; j < col; j++) {
            if (board[j] == i || abs(board[j] - i) == col - j) // the new queue i attacks j
                break;
        }
		if (j < col) continue;

		board[col] = i;
		count += solve(n_queens, col + 1, board);
	}
    return count;
}

int32_t schedule(
        int n_schedule_levels,
        int n_queens,
        int col,
        std::vector<int> board,
        const ExecutionContext& exec_ctx,
        std::vector<AsyncValueRef<int32_t>> &async_vals)
{
    if (col >= n_schedule_levels)
        return solve(n_queens, col, board);
    else {
        for (int i = 0, j = 0; i < n_queens; i++) {
            for (j = 0; j < col; j++) {
                if (board[j] == i || abs(board[j] - i) == col - j) // the new queue i attacks j
                    break;
            }
            if (j < col) continue;

            board[col] = i;
            async_vals.push_back(exec_ctx.host()->EnqueueWork(
                [n_schedule_levels, n_queens, col, board, exec_ctx, &async_vals] {
                    return schedule(n_schedule_levels, n_queens, col + 1, board, exec_ctx, async_vals); }));
        }
    }
    return 0;
}

static int32_t NQueens(int32_t n_queens, int32_t n_schedule_levels,
                                      const ExecutionContext& exec_ctx) {
  size_t max_q_len = 1;
  for (int i = 0; i < n_schedule_levels; ++i)
      max_q_len *= n_queens;

  std::vector<AsyncValueRef<int32_t>> async_vals;
  async_vals.reserve(max_q_len);

  std::vector<int> board(n_queens);
  schedule(n_schedule_levels, n_queens, 0, board, exec_ctx, async_vals);

  int32_t count = 0;
  for (size_t i = 0; i < async_vals.size(); ++i) {
      exec_ctx.host()->Await(async_vals[i].CopyRCRef());
      count += async_vals[i].get();
  }
  return count;
}

void RegisterNQueensKernels(KernelRegistry* registry) {
  registry->AddKernel("nqueens.i32", TFRT_KERNEL(NQueens));
}

}  // namespace tfrt
