#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# 在第一个数之前的字典中查找有没有复合条件的,速度进一步提升

class Solution:
    def twoSum(self, nums, target):
        d = {}
        n = len(nums)
        for x in range(n):
            if target - nums[x] in d:
                return d[target - nums[x]], x
            else:
                d[nums[x]] = x
nums = [2, 7, 11, 15]
target = 9
a = Solution()
res = a.twoSum(nums,9)
print(res)