#!/usr/bin/env python3
# -*- encoding:utf-8 -*-


class Solution:
    def nthUglyNumber(self, n):
        if n <= 0:
            return 0
        ugly = [1]
        i2, i3, i5 = 0, 0, 0
        while n > 1:
            a2, a3, a5 = nums[i2] * 2, nums[i3] * 3, numsp[i5] * 5
            umin = min(a2, a3, a5)
            ugly.append(umin)
            if a2 == umin:
                i2 += 1
            if a3 == umin:
                i3 += 1
            if a5 == umin:
                i5 += 1
            n -= 1

        return ugly[-1]