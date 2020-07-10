#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# dp问题: dp[i] = dp[i>>1] + (i&1)
#
# *  i>>1代表前一个二进制位的次数
# *  i&1代表i的末尾是否为1
def countBits(num):
    dp = [0]
    for i in range(1, num + 1):
        dp.append(dp[i >> 1] + (i & 1))
    return dp

print(countBits(5))

