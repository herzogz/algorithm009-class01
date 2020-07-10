#!/usr/bin/env python3
# -*- encoding:utf-8 -*-


class Solution:
    def totalNQueens(self, n):
        if n < 1:
            return 0
        self.count = 0
        self.dfs(n, 0, 0, 0, 0)
        return self.count

    def dfs(self, n, row, col, pie, na):
        # terminator
        if row >= n:
            self.count += 1
            return
        position = (~(col | pie | na)) & (
            (1 << n) - 1)  # 将最高位为n位清零，整理出皇后可以放的位置
        while position:  # 还有位置可以放皇后的时候
            p = position & -position  # 取最低位的1
            position &= position - 1  # 清除最低位的1， 表示把皇后放过去
            self.dfs(n, row + 1, col | p, (pie | p) << 1, (na | p) >> 1)
