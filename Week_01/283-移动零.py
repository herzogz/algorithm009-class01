#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
"""
思路：
对零计数，在末尾加上0
"""


class Solution:
    def moveZeroes(self, nums):
        count = nums.count(0)
        nums[:] = [i for i in nums if i != 0]
        nums[:] += [0] * count

b = [0, 1, 0, 3, 12]
a = Solution()
a.moveZeroes(b)
print(b)
