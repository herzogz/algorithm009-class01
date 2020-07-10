#!/usr/bin/env python3
# -*- encoding:utf-8 -*-


class Solution:
    def merge(self, nums, start, mid, end):
        l, r = start, mid + 1
        res = []
        while l <= mid and r <= end:
            if nums[l] >= nums[r]:
                res.append(nums[r])
                r += 1
            else:
                res.append(nums[l])
                l += 1
        nums[start:end + 1] = res + nums[l:mid + 1] + nums[r:end + 1]

    def mergesort_and_count(self, nums, start, end):
        if start >= end:
            return 0
        mid = start + ((end - start) >> 2)
        count = self.mergesort_and_count(nums, start, mid) + \
            self.mergesort_and_count(nums, mid + 1, end) # 计数
        j = mid + 1
        for i in range(start, mid + 1):
            while j <= end and nums[i] > 2 * nums[j]: # 比较计数
                j += 1
            count += j - (mid + 1)
        self.merge(nums, start, mid, end)
        return count

    def reversePairs(self, nums):
        return self.mergesort_and_count(nums, 0, len(nums) - 1)
