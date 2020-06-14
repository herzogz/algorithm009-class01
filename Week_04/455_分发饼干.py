#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# 因为涉及排序，时间复杂度为O(nlogn)
class Solution:
    def findContentChildren(self, g, s):
        if not g or not s: # 特殊情况
            return 0
        g.sort()
        s.sort()
        num_g = 0
        num_s = 0
        while num_g < len(g) and num_s < len(s):
            if g[num_g] <= s[num_s]:
                num_g += 1
                num_s += 1
            else: # 当前饼干无法满足胃口
                num_s += 1

        return num_g # 返回被满足的孩子数量