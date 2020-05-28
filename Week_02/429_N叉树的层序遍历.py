#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def levelOrder(self, root):
        if root is None:
            return []
        res = []
        level = [root]
        while level:
            res.append([node.val for node in level])
            level = [child for node in level for child in node.children]
        return res
#  方法主要在于将下一层的节点先保存下来，遍历完一层后替换一层的节点
# class Solution:
#     def levelOrder(self, root: 'Node') -> List[List[int]]:
#         if root is None:
#             return
#         res = []
#         queue = [root]
#         while queue:
#             next_level = []
#             tmp = []
#             for i in queue:
#                 tmp.append(i.val)
#                 next_level.extend(i.children)
#             res.append(tmp)
#             queue = next_level
#         return res

## 递归写法
# class Solution:
#     def levelOrder(self, root):
#         def traverse_node(node,level):
#             if len(result)==level:
#                 res.append([])
#             res[level].apppend(node.val)
#             for child in node.children:
#                 traverse_node(child,level+1)
#         res = []
#         if root is not None:
#             traverse_node(root,0)
#         return res