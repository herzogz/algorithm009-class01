#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# white对应TreeNode数据类型，gray对应int数据类型，所以不需要额外的颜色标记
class Solution:
    def inorderTraversal(self, root):
        stack,res = [root],[]
        while stack:
            node = stack.pop()
            if isinstance(node,TreeNode):
                stack.extend(node.right,node.val,node.left)
            elif isinstance(node,int):
                res.append(node)
        return res