#!/usr/bin/env python3
# -*- encoding:utf-8 -*-

class Solution:
    def jump(self, nums):
        max_position, high_bound, step = 0, 0, 0
        for i in range(len(nums) - 1):  # 去掉最后一位,边界正好为最后一位时的情况会多计算一步
            if max_position >= i:
                max_position = max(max_position, nums[i] + i) # 预先存储好下一个边界
                if i == high_bound: # 到达上一个边界时
                    high_bound = max_position  # 更新新的边界，这一个区间内所能达到的最远距离
                    step += 1 # 有了新的边界，表示会再跨一步
        return step
