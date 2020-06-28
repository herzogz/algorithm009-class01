#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def maximalSquare(self, matrix):
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return 0
        # 最大边长初始化
        maxside = 0
        row = len(matrix)
        column = len(matrix[0])
        dp = [[0] * column for _ in range(row)]
        for i in range(row):
            for j in range(column):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:  # 如果为0行和0列时，dp值最大为1
                        dp[i][j] = 1
                    # 不在第0行和第0列的情况
                    # 当前作为正方形右下角，其最大边长为左，上，左上放最小值+1
                    else:
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1],
                                       dp[i - 1][j - 1]) + 1
                    maxside = max(maxside, dp[i][j])
        return maxside * maxside

