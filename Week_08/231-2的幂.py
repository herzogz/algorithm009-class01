#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def isPowerOfTwo(self, n):
        return (n > 0) and (n & (n - 1) == 0)