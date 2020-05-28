# Hash table

哈希表，也叫散列表，根据**关键码值（key value）**而直接进行访问的数据结构。

它通过把关键码值映射到表中一个位置来访问记录，以加快查找的速度。

这个映射函数叫做散列函数（Hash Function），存放记录的数组叫做哈希表（散列表）

* hash function把要存储的值，映射到一个int的下标
* ![截屏2020-05-25上午11.02.34](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-05-25上午11.02.34.png)

* Hash Collisions

增加维度，同一个位置拉出一个链表（**拉链式解决冲突法**）

如果链表很长，查询效率会退化到O(n)，平均为O(1)

![截屏2020-05-25上午11.05.18](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-05-25上午11.05.18.png)

## 工程实践

* 电话号码簿
* 用户信息表
* 缓存（LRU Cache）
* 键值对存储（Redis）

## python中的Hash table 和Set

* python中对应的Hash table 为字典dict（**key 不重复**）
* set则为集合set（**天然去重**）

 # 树

树（英语：tree）是一种抽象数据类型（ADT）或是实作这种抽象数据类型的数据结构，用来模拟具有树状结构性质的数据集合。它是由n（n>=1）个有限节点组成一个具有层次关系的集合。

如果链表的next指针指向多个节点（升维），那么此时它就变成了树

链表是特殊化的树

树是特殊化的图（没有环的图）

![截屏2020-05-25上午11.18.21](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-05-25上午11.18.21.png)

## 相关术语

- **节点的度**：一个节点含有的子树的个数称为该节点的度；
- **树的度**：一棵树中，最大的节点的度称为树的度；
- **叶节点**或**终端节点**：度为零的节点；
- **父亲节点**或**父节点**：若一个节点含有子节点，则这个节点称为其子节点的父节点；
- **孩子节点或子节点**：一个节点含有的子树的根节点称为该节点的子节点；
- **兄弟节点**：具有相同父节点的节点互称为兄弟节点；
- 节点的**层次**：从根开始定义起，根为第1层，根的子节点为第2层，以此类推；
- 树的**高度**或**深度**：树中节点的最大层次；
- **堂兄弟节点**：父节点在同一层的节点互为堂兄弟；
- **节点的祖先**：从根到该节点所经分支上的所有节点；
- **子孙**：以某节点为根的子树中任一节点都称为该节点的子孙。
- **森林**：由m（m>=0）棵互不相交的树的集合称为森林；



## 树的种类

- **无序树**：树中任意节点的子节点之间没有顺序关系，这种树称为无序树，也称为**自由树**；

- **有序树**

  ：树中任意节点的子节点之间有顺序关系，这种树称为**有序树**；

  - 二叉树

    ：每个节点最多含有两个子树的树称为二叉树；

    - **完全二叉树**：对于一颗二叉树，假设其深度为d(d>1)。除了第d层外，其它各层的节点数目均已达最大值，且第d层所有节点从左向右连续地紧密排列，这样的二叉树被称为完全二叉树，
      - 完全二叉树的特点是：“叶子节点的位置比较规律”。因此在对数据进行**排序或者查找**时可以用到它，比如**堆排序**就使用了它
      - **满二叉树**的定义是所有叶节点都在最底层的完全二叉树;
    - **平衡二叉树**（AVL树）：是一种二叉排序树，要么它是一棵空树，要么它的左子树和右子树都是平衡二叉树，且左子树和右子树的深度之差的绝对值不超过1。

    - **排序二叉树**（二叉查找树（英语：Binary Search Tree），也称二叉搜索树、有序二叉树）；
    - [理解完全二叉树、平衡二叉树、二叉查找树](https://juejin.im/entry/5afb9fb66fb9a07ab458cc0d)

  - **霍夫曼树**（用于信息编码）：带权路径最短的二叉树称为哈夫曼树或最优二叉树；

  - **B树**：一种对读写操作进行优化的自平衡的二叉查找树，能够保持数据有序，拥有多余两个子树。

## 二叉树

* 在二叉树的第i层上至多有2^(i-1)个结点（i>0）
* 深度为k的二叉树至多有2^k - 1个结点（k>0）
* 对于任意一棵二叉树，如果其叶结点数为N0，而度数为2的结点总数为N2，则**N0=N2+1**;
* 具有n个结点的完全二叉树的深度必为 log2(n+1)
* 对完全二叉树，若从上至下、从左至右编号，则编号为i 的结点，其左孩子编号必为2i，其右孩子编号必为2i＋1；其双亲的编号必为i/2（i＝1 时为根,除外）

![截屏2020-05-25上午11.20.51](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-05-25上午11.20.51.png)

## 图

有环的树

![截屏2020-05-25上午11.21.46](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-05-25上午11.21.46.png)

## 树节点的实现

```python
class TreeNode:
  def __init__(self,val):
    self.val = val
    self.left,self.rigtht = None,None
```

## 为什么有树这个结构

* 在斐波拉契数列递归求解时，扩散出去的节点为一个树状节点（**递归树**）
* 下期时，每一步往后走棋盘的状态都会向外扩散成不同的状态，形成一个树状的结构，不同的棋它的树状空间（状态树空间），决策树空间（博弈的空间）复杂度不一样从而决定了游戏或者棋的复杂度不一样
* 人生状态类似于树结构，不同的选择分叉出去不同更多别的选择（芒格所说：后果的后果）

# 二叉树的遍历

* 前中后就是查询时根的先后顺序，查询三句语言的不同顺序
* 前序（Pre-order): 根-左-右
* 中序（In-order):左-根-右
* 后续（Post-order)：左-右-根

树的结构写循环遍历相对实现困，所以使用递归进行遍历

```python
def preorder(self,root):
  if root:
    self.traverse_path.append(root.val)
    self.preorder(root.left)
    self.preorder(root.right)
    
def inorder(self,root):
  if root:
    self.inorder(root.left)
    self.traverse_path.append(root.val)
    self.inorder(root.right)
    
def postorder(self,root):
  if root:
    self.postorder(root.left)
    self.postorder(rott.rigtht)
    self.traverse_path.append(root.val)
```

# 二叉搜索树  Binary Search Tree

二叉搜索树（**BST**），也叫二叉排序树，有序二叉树（Ordered Binary Tree)，排序二叉树（Sorted Binary Tree)，是指一棵**空树**或者具有下列性质的二叉树：

* 左子树上**所有结点**的值均小于它的根节点的值
* 右子树上**所有结点**的值均大于它的根节点的值
* 以此类推：左，右子树也分别为二叉查找树（重复性）

***中序遍历：升序排列***



## 常见操作

1. 查询 O(logn) （向左走还是向右走）
   1. 类似于二分法，每次查询时，同根节点比较，如果小于根节点则只能在左边（右边同理），每次可以筛掉一半的结点。
2. 插入新结点（创建）O(logn)
   1. 每次查找，如果没有查找到的话，当前位置应该就是结点插入位置
   2. 一棵空树就是二叉搜索树，将后续结点依次插入到空树中（重复上面的操作）即可
3. 删除 O(logn)
   1. 如果是叶结点，直接删除
   2. 删除结点时，一般寻找右子树中最接近根节点的结点（**右子树中最小的结点**）替换掉被删除的结点

特殊情况：退化成了链表，查找时间复杂度为 O(n)

# 常见的一些树的应用场景

* xml，html等，那么编写这些东西的解析器的时候，不可避免用到树
* 路由协议就是使用了树的算法
* MySqll数据库索引
* 文件系统的目录结构
* AI算法:决策树

# 堆

**Heap**：可以迅速找到一堆数中的最大或者最小值的数据结构，是一种**完全二叉树**

我们将将根节点最大的堆叫做**大顶堆**或者**大根堆**，根节点最小的堆叫做**小顶堆**

![截屏2020-05-27下午9.15.00](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-05-27下午9.15.00.png)

常见的堆有**二叉堆**，**斐波拉契堆**等。(实现一般用二叉堆，斐波拉契堆空间复杂度更好，多叉树)

假设是大顶堆，常见操作（API）：

find-max O(1) （合格的标准）

delete-max O(logN)

insert(create) O(logN) or O(1)

[维基百科：堆（Heap）](https://en.wikipedia.org/wiki/Heap_(data_structure))

## 二叉堆的性质

通过**完全二叉树**来实现（如果用二叉搜索树就会变慢，因为查找会变成O(logN)）

二叉堆（大顶）满足下列性质：

* 是一棵完全树
* 树中任意结点的值总是 >= 其子节点的值（保证根节点为最大值）

![截屏2020-05-26下午9.30.37](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-05-26下午9.30.37.png)

## 二叉堆的实现

1. 二叉堆一般通过**数组**实现
2. 假设**第一个元素**在数组中的索引为0的话，则副节点和子节点的位置关系如下：
   1. 根节点（顶堆元素）是：a[0]
   2. 索引为i的左孩子的索引是（2*i+1）
   3. 索引为i的右孩子索引为（2*i+2）
   4. 索引为i的副节点的索引是floor（（i-1）/2）
   5. 如果left，right的值超出了数组的索引，则表示这个节点是不存在的

![截屏2020-05-27下午9.17.42](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-05-27下午9.17.42.png)

## insert 插入操作 O(log N)

1. 新元素一律先插入到堆的尾部

2. 依次向上递归调整整个堆的结构（一直到根）

**HeapifyUp**

最坏时间复杂度O(log N)  树的深度

## delete max 删除顶堆操作 O(log N)

1. 将堆尾元素替换到顶部
2. 依次从根部向下调整整个堆的结构（一直到堆尾即可）
   1. 比较它与它的两个子节点中三个值的大小，选择最大的值放到父节点上
   2. 使用递归的方式向下比较

**HeapifyDown**

![截屏2020-05-26下午10.02.54](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-05-26下午10.02.54.png)

## 注意

二叉堆事堆（优先队列 priority_queue）的一种常见简单的实现，**并不是最优解**

## python中的heapq模块

python模块heapq中的堆是一个最小堆，索引从0开始，其排序是不稳定的。

Python中创建一个堆可以直接使用list的创建方式H = [], 或者使用heapify()函数将一个存在的列表转为堆。

### 基本操作

`heapq.heappush(heap,item)`

将 *item* 的值加入 *heap* 中，保持堆的不变性。

`heapq.heappop(heap)`

弹出并返回 *heap* 的最小的元素，保持堆的不变性。使用 `heap[0]` ，可以只访问最小的元素而不弹出它。

`heapq.heappushpop(heap,item)`

将 *item* 放入堆中，然后弹出并返回 *heap* 的最小元素。该组合操作比先调用 heappush()再调用 heappop()运行起来更有效率。

`heapq.heapify(x)`

将list *x* 转换成堆，原地，线性时间内。

```python
In [2]: import heapq

In [3]: a = [1,2,4,6,7,7,4,3,2,1,0,65]

In [4]: heapq.heapify(a)

In [5]: a
Out[5]: [0, 1, 4, 2, 1, 7, 4, 3, 6, 2, 7, 65]
```

`heapq.heapreplace(heap, item)`

弹出并返回 *heap* 中最小的一项，同时推入新的 *item*。 堆的大小不变。

* `heapq.merge`(**iterables*, *key=None*, *reverse=False*)

将多个已排序的输入合并为一个已排序的输出（例如，合并来自多个日志文件的带时间戳的条目）。 返回已排序值的 iterator。

类似于 `sorted(itertools.chain(*iterables))` 但返回一个可迭代对象，不会一次性地将数据全部放入内存，并假定每个输入流都是已排序的（从小到大）。

* `heapq.nlargest`(*n*, *iterable*, *key=None*)

  从 *iterable* 所定义的数据集中返回前 *n* 个最大元素组成的列表。 如果提供了 *key* 则其应指定一个单参数的函数，用于从 *iterable* 的每个元素中提取比较键 (例如 `key=str.lower`)。

* `heapq.nsmallest`(*n*, *iterable*, *key=None*)

后两个函数在 *n* 值较小时性能最好。 对于更大的值，使用 sorted() 函数会更有效率。 此外，当 `n==1` 时，使用内置的 `min()`和 `max()` 函数会更有效率。 如果需要重复使用这些函数，请考虑将可迭代对象转为真正的堆。

```python
import heapq
nums = [10, 2, 9, 100, 80]
print(heapq.nlargest(3, nums))
print(heapq.nsmallest(3, nums))

students = [{'names': 'cc', 'score': 100, 'height': 189},
            {'names': 'bb', 'score': 10, 'height': 169},
            {'names': 'dd', 'score': 80, 'height': 179}]
print(heapq.nlargest(2, students, key=lambda x: x['height']))  # 相当好用

[100, 80, 10]
[2, 9, 10]
[{'names': 'cc', 'score': 100, 'height': 189}, {'names': 'dd', 'score': 80, 'height': 179}]
```



# 图



图的相关算法和应用较少，工程中结合实际业务来考虑

![截屏2020-05-26下午10.14.09](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-05-26下午10.14.09.png)

## 图的属性和分类

### 图的属性

* Graph（V，E）
* V-vertex：点
  * 度—入度和出度：表示一个点有多少个边
  * 点与点之间：连通与否
* E-edge： 边
  * 有向和无向
  * 权重（边长）

**无向无向图**

邻接矩阵为对称矩阵

![截屏2020-05-26下午10.19.25](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-05-26下午10.19.25.png)

**有向无权图**

![截屏2020-05-26下午10.22.19](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-05-26下午10.22.19.png)

**无向有权图**

![截屏2020-05-26下午10.23.21](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-05-26下午10.23.21.png)

## 基于图的相关算法(代码模版记下来）

* DFS-递归写法
* 不要忘记加visited集合（树可以保证没有环路，图可能会有环路重复）

```python
visited = set() # 和树中的DFS的最大区别

def dfs(node,visited):
  if node in visited: 
    # already visited
    return
  
  	visited.add(node)
    
    #process current node here.
    ...
    for next_node in node.children():
      if not next_node in visited:
        dfs(next_node,visited)
```

* BFS代码

```python
def BFS(graph,start,end):
  queue = []
  queue.append([start])
  
  visited = set() 
  while queue:
    node = queue.pop()
    visited.add(node)
    
    process(node)
    nodes = generate_related_nodes(node)
    queue.push(nodes)
```

## 图的高级算法

1. [连通图个数](https://leetcode-cn.com/problems/number-of-islands/)
2. [拓扑排序Topological Sorting]([ https://zhuanlan.zhihu.com/p/34871092](https://zhuanlan.zhihu.com/p/34871092))
3. [最短路径（Shortest Path)Dijkstra](https://www.bilibili.com/video/av25829980?from=search&seid=13391343514095937158)
4. [最小生成树（Minimum Spanning Tree)](https://www.bilibili.com/video/av84820276?from=search&seid=17476598104352152051)



