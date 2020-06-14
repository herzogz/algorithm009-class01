#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# 关键点在于从后向前贪心，依次更新可以到后一个位置的索引值
class Solution:
    def canJump(self, nums):
        if not nums:
            return
        if 0 not in nums[:-1]: # 如果所有元素都不为0，那么一定可以走到最后
            return True
        end = len(nums) - 1
        for i in range(end - 1, -1, -1): # 倒着遍历，从倒数第二个数开始看
            if nums[i] + i >= end:
                end = i
        return end == 0
# 正向
# class Solution:
#     def canJump(self, nums) :
#         max_i = 0       #初始化最远的位置
#         for i, num in enumerate(nums):
#             if max_i >= i and i+num > max_i:  # 若当前位置+跳数>最远位置
#                 max_i = i+num  #更新
#         return max_i>=i # 此时i为最后一位
