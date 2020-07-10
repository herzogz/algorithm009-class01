#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def reverseBits(self, n):
        res = 0
        for _ in range(32):
            res = (res<<1) + (n&1) # 最后一位挪到前面去
            n >>= 1 再把最后一位去掉
        return res


# class Solution:
#     def reverseBits(self, n):
#         ans, mask = 0, 1
#         for i in range(32):
#             if n & mask:
#                 ans |= 1 << (31 - i)  # 置1操作
#             mask <<= 1
#         return ans

