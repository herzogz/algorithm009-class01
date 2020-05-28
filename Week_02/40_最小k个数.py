#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
import heapq
class Solution:
     def getLeastNumbers(self, arr, k):
         if k == 0:
             return []
         if len(arr)<= k:
             return arr
         hp = [-x for x in arr[:k]] # 取负数，小根堆变大根堆
         heapq.heapify(hp)
         for i in range(k,len(arr)):
             if -hp[0]>arr[i]: # 如果堆顶大于这个数
                 heapq.heappop(hp)
                 heapq.heappush(hp,arr[i])
         res = [-x for x in hp]

         return res



# 排序取数
# class Solution:
#     def getLeastNumbers(self, arr, k):
#         arr.sort()
#         return arr[:k]