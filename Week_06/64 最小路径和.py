#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def minPathSum(self, grid):
        if not grid:
            return 0
        for m in range(len(grid)):
            for n in range(len(grid[0])):
                if m == n == 0:
                    continue
                elif m == 0 : # 第一行
                    grid[m][n] += grid[m][n-1]
                elif n == 0: # 第一列
                    grid[m][n] += grid[m-1][n]
                else: # 中间部分
                    grid[m][n] += min(grid[m-1][n],grid[m][n-1])
        return grid[-1][-1]