#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution(object):
    def isIsomorphic(self, s, t):

        if not s:
            return True
        dic={}
        for i in range(len(s)):
            if s[i] not in dic: # 如果不在字典中
                if t[i] in dic.values(): # 另一个字符的值却在字典中
                    return False
                else:
                    dic[s[i]]=t[i] # 一一对应起来
            else:
                if dic[s[i]]!=t[i]:# 发现不对应的字符
                    return False
        return True