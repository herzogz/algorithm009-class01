#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def firstUniqChar(self, s) :
        if not s:
            return -1
        hashmap = collections.defaultdict(int)
        for char in s:
            hashmap[char] += 1
        for char in hashmap:
            if hashmap[char] == 1:
                return s.index(char)
        return -1