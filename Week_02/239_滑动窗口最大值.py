#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
import heapq
class Solution:
    def maxSlidingWindow(self, nums,k):
        res = []
        hp = []
        for i in range(len(nums)):
            heapq.heappush(hp,(-nums[i],i))
            if i > k-1: # 窗口取满后
                while hp and hp[0][1]< i+1-k:# 堆中最大值已不在窗口中
                    heapq.heappop(hp)
                res.append(-hp[0][0])
        return res



