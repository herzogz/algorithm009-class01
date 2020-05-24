#!/usr/bin/env python3
# -*- encoding:utf-8 -*-


class Solution:
    def mergeTwoLists(self, l1, l2) -> ListNode:
        dummy = ListNode(0)  # 创建哑节点
        move = dummy  # 指针指向哑节点

        # 当两个链表都没有遍历完成时
        while l1 and l2:
            if l1.val < l2.val:
                move.next = l1
                l1 = l1.next  # 链表移动到下个节点
            else:
                move.next = l2
                l2 = l2.next
            move = move.next  # move永远指向末端
        move.next = l1 if l1 else l2  # 拼接剩余元素
        return dummy.next
