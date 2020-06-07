#!/usr/bin/env python3
# -*- encoding:utf-8 -*-
#如果根节点大于p和q的值，则在左节点寻找
#如果根节点小于p和q的值，则在右节点寻找
#其他情况，说明节点p和q一个在左子树， 一个在右子树

class Solution:
    def lowestCommonAncestor(self, root, p, q):
        if not root:
            return None
        if root == p or root == q:
            return root
        if root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root
