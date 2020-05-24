#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# nums[:]并没有开辟新的空间
class Solution:
    def rotate(self, nums, k) -> None:
        nums[:] = nums[len(nums)-k:]+nums[:len(nums)-k]

nums = [1,2,3,4,5,6,7]
k = 3
a = Solution()
a.rotate(nums,k)
print(nums)