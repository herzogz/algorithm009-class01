#!/usr/bin/env python3
# -*- encoding:utf-8 -*-


class Solution:
    def reverseWords(self, s: str) -> str:
        ls = s.split()
        return ' '.join([i[::-1] for i in ls])
