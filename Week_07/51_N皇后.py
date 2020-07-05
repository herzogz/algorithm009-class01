#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def solveNQueens(self, n):
        def DFS(queens, left_diagnal, right_diagnal):
            row = len(queens)
            if row == n:  # 皇后放满了
                res.append(queens)

            for col in range(n):
                if col not in queens and row + col not in left_diagnal \
                        and row - col not in right_diagnal:
                    DFS(queens + [col], left_diagnal + [row + col],
                        right_diagnal + [row - col])

        res = []
        DFS([], [], [])
        return [['.' * i + 'Q' + '.' * (n - i - 1) for i in cols] for cols in
                res]

