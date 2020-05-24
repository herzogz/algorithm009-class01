#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
"""
思路：

1. 当前指针下标1开始，当下标小于数组长度时，遍历
2. 如果nums[index]== nums[index-1],则删除nums[index]
3. 不同则index +1
"""
class Solution:
    def removeDuplicates(self, nums) -> int:
        cur = 1
        while cur < len(nums):
            if nums[cur]==nums[cur-1]:
                nums.pop(cur)
            else:
                cur += 1
        return len(nums)

l = [1,2,3,4,4,5,6,7,7,8]
a = Solution()
a.removeDuplicates(l)
print(l)