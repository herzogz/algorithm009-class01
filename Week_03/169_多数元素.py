#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# from collections import Counter
# class Solution:
#     def majorityElement(self, nums):
#         return sorted(Counter(nums).items(), key = lambda x:x[1])[-1][0]
# 分治
class Solution:
    def majorityElement(self, nums) :
        return self.find_mode(nums)

    def find_mode(self, nums):
        if len(nums) == 1: return nums[0]
        mid = len(nums) // 2
        left_num = self.find_mode(nums[0:mid])
        right_num = self.find_mode(nums[mid:])
        if left_num != right_num:
            return left_num if nums.count(left_num) >= nums.count(
                right_num) else right_num
        else:
            return left_num
