#!/usr/bin/env python3
# -*- encoding:utf-8 -*-


class Solution:
    def reverseWords(self, s) :
        i = len(s) - 1
        tmp = ''
        while i >= 0:  # 翻转全部字符串
            tmp += s[i]
            i -= 1
        # 翻转单词
        return ' '.join([i[::-1] for i in tmp.split()])
