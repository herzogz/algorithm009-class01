#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
"""
* 利用左右指针的下标差值计算出每一行雨水+柱子的体积，如图第一行体积为11，第二行为8，第三行为1。累加得到整体体积tmp=20tmp=20（每一层从左边第一个方格到右边最后一个方格之间一定是被蓝黑两种颜色的方格填满的，不会存在空白，这也是为什么能按层求的关键）
* 计算柱子体积，为height：[0,1,0,2,1,0,1,3,2,1,2,1]height：[0,1,0,2,1,0,1,3,2,1,2,1] 数组之和SUM=14SUM=14（也可以直接用sum()函数，不过时间复杂度就是O(2n)了）
* 返回结果 tmp−SUM就是雨水的体积
"""


class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        left, right = 0, n - 1
        total, temp, height_ = 0, 0, 1
        while left <= right:
            while left <= right and height[left] < height_:
                total += height[left]
                left += 1
            while left <= right and height[right] < height_:
                total += height[right]
                right -= 1
            height_ += 1
            temp += right - left + 1  # 记得加1
        return temp - total
