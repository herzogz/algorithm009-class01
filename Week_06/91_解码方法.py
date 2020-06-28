#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# 类似于爬楼梯问题
# dp[i] = dp[i-1] + dp[i-2]
class Solution:
    def numDecodings(self, s):
        if s[0] == '0' or not s: # 不能解码的情况
            return 0

        pre, cur = 1, 1
        for i in range(1, len(s)):
            # 当前的方法 = 上上次的方法可以跨两步 + 上次的方法可以跨一步
            pre, cur = cur, pre * (9 < int(s[i-1:i+1]) <= 26) + cur * (int(s[i]) > 0)
        return cur