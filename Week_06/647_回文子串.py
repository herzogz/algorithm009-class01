#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def countSubstrings(self, s):
        if not s:
            return 0
        n = len(s)
        # 定义dp
        dp = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            dp[i][i] = 1  # 单个字符一定是回文字符
        res = n
        # [k,j] 之间的回文字符串情况
        for j in range(n):
            for k in range(0, j):
                # 对角线情况之前考虑过了

                if j - k == 1:  # 相邻两个数是回文字符串的情况
                    if s[j] == s[k]:
                        dp[k][j] = 1
                        res += 1
                elif j - k > 1:
                    if s[j] == s[k]:  # 超过2个数的情况，是否是回文字符串取决于去掉两边以后中间字符串是否是回文字符串
                        dp[k][j] = dp[k + 1][j - 1]
                        if dp[k][j] == 1:
                            res += 1
        return res
#        res = 0
#
#         for i in range(n):
#             for j in range(i,n):
#                 if dp[i][j] == 1:
#                     res += 1
#         return res
test = Solution()
a = 'aaa'
print(test.countSubstrings(a))