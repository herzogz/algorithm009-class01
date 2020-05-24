#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
"""
* 倒序遍历，判断每一位是否为9
* 如果不为9，则不发生进位，加1后直接返回
* 如果为9，则设着为0，向前遍历（此时带着一个进位1）
* 如果下一位仍为0，则继续带着进位往前遍历，以此类推
* 如果遍历结束所有位都为9，则在首位插入1
"""


class Solution:
    def plusOne(self, digits):
        n = len(digits)
        i = n - 1
        while i >= 0:
            if digits[i] != 9:
                digits[i] += 1
                return digits
            digits[i] = 0
            i -= 1
        digits.insert(0, 1)
        return digits
a = Solution()
digits = [1,2,3]
print(a.plusOne(digits))