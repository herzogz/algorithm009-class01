#!/usr/bin/env python3
# -*- encoding:utf-8 -*-


class Solution:
    def reverseStr(self, s, k):
        result = ''
        for i in range(0, len(s), 2 * k):
            tmp = s[i:i + k]  # k个数字反转
            tmp = tmp[::-1] + s[i + k:i + 2 * k]  # 加上没有翻转的部分
            result += tmp
        return result
