#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
class Solution:
    def solveSudoku(self, board):
        """
        Do not return anything, modify board in-place instead.
        """
        # 初始化出每个区间可用的数字
        row = [set(range(1, 10)) for _ in range(9)]
        column = [set(range(1, 10)) for _ in range(9)]
        block = [set(range(1, 10)) for _ in range(9)]

        empty = []  # 收集需要填数的位置

        # 清除已经使用的数字
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    number = int(board[i][j])
                    row[i].remove(number)
                    column[j].remove(number)
                    block[(i // 3) * 3 + j // 3].remove(number)
                else:
                    empty.append((i, j))  # 添加填数的位置

        def backtrack(level=0):
            if level == len(empty):  # 表示empty中所有位置处理完了
                return True
            i, j = empty[level]
            b = (i // 3) * 3 + j // 3
            for val in row[i] & column[j] & block[b]:  # 遍历剩余可用的数字
                row[i].remove(val)
                column[j].remove(val)
                block[b].remove(val)
                board[i][j] = str(val)
                if backtrack(level + 1):
                    return True
                # 如果行不通，恢复现场
                row[i].add(val)
                column[j].add(val)
                block[b].add(val)
            return False

        backtrack()