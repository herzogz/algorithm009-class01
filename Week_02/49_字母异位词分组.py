#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
from collections import defaultdict


class Solution:
    def groupAnagrams(self, strs):
        dic = defaultdict(list)
        for i in strs:
            # 注意sorted会把字符串拆成列表，所以需要重新连成字符串作为key
            dic[''.join(sorted(i))].append(i)
        return list(dic.values())


a = Solution()
s = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(a.groupAnagrams(s))
