#!/usr/bin/env python3
# -*- encoding:utf-8 -*-


class Solution:
    def reverseOnlyLetters(self, S: str) -> str:
        if not S:
            return ''
        S = list(S)
        front = 0
        rear = len(S) - 1
        while front < rear:
            if not S[front].isalpha():
                front += 1
            elif not S[rear].isalpha():
                rear -= 1
            else:
                S[front], S[rear] = S[rear], S[front]
                front += 1
                rear -= 1
        return ''.join(S)
