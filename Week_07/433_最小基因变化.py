#!/usr/bin/env python3
# -*- encoding:utf-8 -*-

class Solution:
    def minMutation(self, start , end, bank):
        bank = set(bank)

        if not end or not start or end not in bank:
            return -1

        change = {
            'A': "CGT",
            'C': "AGT",
            'G': 'ACT',
            'T': 'ACG'
        }
        queue = [(start, 0)]

        while queue:
            node, step = queue.pop(0)
            if node == end:
                return step

            for i, v in enumerate(node):
                for c in change[v]:
                    new = node[:i] + c + node[i + 1:]

                    if new in bank:
                        queue.append((new, step + 1))
                        bank.remove(new)
        return -1

# class Solution:
#     def minMutation(self, start: str, end: str, bank: List[str]) -> int:
#         bank = set(bank)

#         if not end or not start or end not in bank:
#             return -1

#         change = {
#             'A':"CGT",
#             'C':"AGT",
#             'G':'ACT',
#             'T':'ACG'
#         }

#         def dfs(node,count,_bank):
#             # terminator
#             if node == end:
#                 counts.append(count)
#                 return
#             if not _bank:
#                 return
#             # process
#             for i ,v in enumerate(node):
#                 for c in change[v]:
#                     new = node[:i] + c + node[i+1:]
#                     if new in _bank:
#                         _bank.remove(new)
#                         # drill down
#                         dfs(new,count+1,_bank)

#                         _bank.add(new)
#         counts = []
#         dfs(start,0,bank)
#         if not counts:
#             return -1
#         return min(counts)



