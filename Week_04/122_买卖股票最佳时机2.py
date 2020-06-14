#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# 只要后一天比前一天大，则前一天买，后一天抛，即可锁住利润

class Solution:
    def maxProfit(self, prices):
        profit = 0
        for i in range(1, len(prices)):
            profit += prices[i] - prices[i - 1] if prices[i] > prices[i - 1] else 0
        return profit
