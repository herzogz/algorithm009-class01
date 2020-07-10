#!/usr/bin/env python3
# -*- encoding:utf-8 -*-


class LRUCache:

    def __init__(self, capacity):
        self.dic = collections.OrderedDict()
        self.remain = capacity

    def get(self, key):
        if key not in self.dic:
            return -1
        v = self.dic.pop(key)
        self.dic[key] = v
        return v

    def put(self, key, value):
        if key in self.dic:
            self.dic.pop(key)  # 先弹出
        else:
            if self.remain > 0:
                self.remain -= 1
            else:
                self.dic.popitem(last=False)
        self.dic[key] = value  # 放到最前面
