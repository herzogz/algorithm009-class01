#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
"""
* 由于nums1队尾已有0元素作为留给nums2的空间，所以我们可以从对尾将nums2插入
* 设指针p指向nums1队尾，标记待添加元素的位置
* 指针p1指向nums1元素尾m-1,p2指向nums2队尾n-1
* 当p1，p2>=0时，比较p1,p2所指元素，更小的数值放在p的位置，并移动更小数值对应的坐标，p坐标向前移动
* 最后p1,p2<0时，将剩余nums2加入(若nums2先添加完则不用添加)
"""


class Solution:
    def merge(self, nums1, m, nums2,n) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1 = m - 1
        p2 = n - 1
        p = m + n - 1

        while p1 >= 0 and p2 >= 0:
            if nums1[p1] < nums2[p2]:
                nums1[p] = nums2[p2]
                p2 -= 1
            else:
                nums1[p] = nums1[p1]
                p1 -= 1
            p -= 1
        nums1[:p2 + 1] = nums2[:p2 + 1]

a = Solution()
b = [1,2,4,0,0,0]
c = [1,6,8]
a.merge(b,3,c,3)
print(b)