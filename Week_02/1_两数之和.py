#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def twoSum(self,nums,target):
        hashmap = {}
        for i in range(len(nums)):
            if target-nums[i] in hashmap:
                return [i,hashmap[target-nums[i]]]
            else:
                hashmap[nums[i]] = i

a = Solution()
nums = [2, 7, 11, 15]
target = 9
print(a.twoSum(nums,target))
