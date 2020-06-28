#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def leastInterval(self, tasks, n) :

# 最短时间需要把各种不同的任务排在一起执行
# 先排列好任务数最多的任务，再把其他少于最多任务的数的任务循环插入再间隔时间中
        count = [0] * 26
        for ch in tasks:
            count[ord(ch) - ord('A')] += 1

        count_max = max(count)
        total = (count_max - 1) * (n + 1)

        for _ in count:
            if _ == count_max:
                total += 1

        return max(total, len(tasks))
