#!/usr/bin/env python3
# -*- encoding:utf-8 -*-


class Solution:
    def relativeSortArray(self, arr1, arr2):
        bins = [0 for _ in range(1001)]
        res = []
        for i in arr1:
            bins[i] += 1
        for i in arr2:
            res += [i] * bins[i]
            bins[i] = 0
        for i in range(len(bins)):
            res += [i] * bins[i]  # 其他所有位皆为0，只有还剩下没有出现过的元素按顺序添加

        return res
