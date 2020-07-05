#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def findWords(self, board, words):
        if not board or not board[0]:
            return []
        if not words:
            return []
        # 构建Trie
        root = {}
        for word in words:
            node = root
            for char in word:
                node = node.setdefault(char, {})
            node['#'] = True

        # DFS
        def dfs(i, j, cur_node, pre, visited):
            # terminator
            if '#' in cur_node:
                res.add(pre)
            for (dx, dy) in ((-1, 0), (0, 1), (0, -1), (1, 0)):
                x = i + dx
                y = j + dy
                if -1 < x < m and -1 < y < n and \
                        board[x][y] in cur_node and (x, y) not in visited:
                    dfs(x, y, cur_node[board[x][y]], pre + board[x][y],
                        visited | {(x, y)})

        res = set()
        m, n = len(board), len(board[0])
        for i in range(m):
            for j in range(n):
                if board[i][j] in root:
                    dfs(i, j, root[board[i][j]], board[i][j], {(i, j)})
        return list(res)