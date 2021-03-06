#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution(object):
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        isPalindrome = lambda s: s == s[::-1]
        strPart = lambda s, x: s[:x] + s[x + 1:]
        left = 0
        right = len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return isPalindrome(strPart(s, left)) or isPalindrome(strPart(s, right))
            left += 1
            right -= 1
        return True

# class Solution(object):
#     def validPalindrome(self, s):
#         """
#         :type s: str
#         :rtype: bool
#         """
#         isPalindrome = lambda x : x == x[::-1]
#         left, right = 0, len(s) - 1
#         while left <= right:
#             if s[left] == s[right]:
#                 left += 1
#                 right -= 1
#             else:
#                 return isPalindrome(s[left + 1 : right + 1]) or isPalindrome(s[left: right])
#         return True