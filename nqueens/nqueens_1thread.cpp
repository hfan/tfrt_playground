#include <iostream>
#include <vector>
#include <stdlib.h>

#include "Timer.h"

uint64_t count = 0;

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
 
int main(int argc, char **argv)
{
    int n_queens = 0;
    if (argc <= 1 || (n_queens = atoi(argv[1])) <= 0) {
        std::cerr << "Usage: " << argv[0] << " <n_queens>" << std::endl;
        return 1;
    }

    std::vector<int> board(n_queens);

    Timer timer;
    timer.start_timer();
	solve(n_queens, 0, board);
    timer.end_timer();

    std::cout << "Num of solutions: " << count << std::endl;
    std::cout << "Time = " << timer.get_timed_interval() << " us" << std::endl;
}
