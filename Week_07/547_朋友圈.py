#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def findCircleNum(self, M):
        if not M:
            return 0

        # 初始化并查集
        n = len(M)
        p = [i for i in range(n)]

        def _union(p, i, j):
            p1 = _parent(p, i)
            p2 = _parent(p, j)
            p[p1] = p2

        def _parent(p, i):
            root = i
            while p[root] != root:  # 查询代表元素root
                root = p[root]
            while p[i] != i:
                x = i
                i = p[i]
                p[x] = root  # 把所有元素指向root，压缩路径
            return root

        for i in range(n):
            for j in range(n):
                if M[i][j] == 1:
                    _union(p, i, j)
        return len(set(_parent(p, i) for i in range(n)))