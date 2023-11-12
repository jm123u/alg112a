#include<stdio.h>
#include<stdlib.h>
#include<stdbool.h>

#define N 8

bool is_safe(int board[N], int row, int col) {
    for (int i = 0; i < row; i++) {
        if (board[i] == col || abs(board[i] - col) == abs(i - row)) {
            return false;
        }
    }
    return true;
}

void print_solution(int board[N]) {
    for (int i = 0; i < N; i++, puts(""))
        for (int j = 0; j < N; j++)
            putchar(board[i] == j ? 'Q' : '.');
    puts("");
}

void solve_eight_queens(int board[N], int row) {
    if (row == N) {
        print_solution(board);
        return;
    }
    for (int col = 0; col < N; col++) {
        if (is_safe(board, row, col)) {
            board[row] = col;
            solve_eight_queens(board, row + 1);
            board[row] = -1;
        }
    }
}

int main() {
    int board[N];
    for (int i = 0; i < N; i++) {
        board[i] = -1;
    }
    solve_eight_queens(board, 0);

    return 0;
}
