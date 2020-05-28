#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def preorder(self, root):
        if root is None:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.children:
                for c in children[::-1]:
                    satck.append(c)
        return res