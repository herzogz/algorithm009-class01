#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
# from collections import Counter
# import heapq
# class Solution:
#     def topKFrequent(self, nums, k):
#         count = collections.Counter(nums)
#         return heapq.nlargest(k, count.keys(), key=count.get)
#         # count.get返回key对应的value，这里用value来排序返回最大的key

from collections import defaultdict
import heapq
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        res = []
        hash_map = defaultdict(int)
        hp = []
        for i in nums:
            hash_map[i] += 1
        for i in hash_map:
            heapq.heappush(hp,(-hash_map[i],i))
        for j in range(k):
            res.append(heapq.heappop(hp)[1]) # 返回最高频元素的key值

        return res