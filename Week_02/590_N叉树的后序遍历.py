#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        if root is None:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            for c in node.children:
                    if c:
                        stack.append(child)
        return res[::-1]