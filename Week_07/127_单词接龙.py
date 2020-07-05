#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
import string


class Solution:
    def ladderLength(self, beginWord, endWord,
                     wordList):
        if not beginWord or not endWord or not wordList:
            return 0
        if endWord not in wordList:
            return 0

        step = 1
        begin = {beginWord}
        end = {endWord}

        n = len(beginWord)
        wordlist = set(wordList)

        while begin:
            step += 1
            new_begin = set()
            for word in begin:
                for i in range(n):
                    for char in string.ascii_lowercase:
                        if char != word[i]:
                            new_word = word[:i] + char + word[i + 1:]
                            if new_word in end:  # 与反向的扩散相遇
                                return step
                            if new_word in wordlist:
                                new_begin.add(new_word)
                                wordlist.remove(new_word)
            begin = new_begin
            if len(end) < len(begin):  # 交换方向，更小的优先搜索
                begin, end = end, begin
        return 0