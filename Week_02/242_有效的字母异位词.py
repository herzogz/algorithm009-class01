#!/usr/bin/env python3
# # -*- encoding:utf-8 -*-
from collections import defaultdict
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        counter = defaultdict(lambda : 0)
        for i in s:
            counter[i] += 1
        for k , v in counter.items():
            if t.count(k) != v:
                return False
        return True
# class Solution:
#     def isAnagram(self, s: str, t: str) -> bool:
#         if len(s) != len(t):
#           return False
#         se = set(s)
#         if se == set(t):
#             for i in se:
#                 # 直接比较字符元素个数比较字符的个数
#                 if s.count(i) != t.count(i):
#                   return False
#             return True
#         else:
#             return False




s = 'anagram'
t = 'nagaram'
a = Solution()
print(a.isAnagram(s, t))
