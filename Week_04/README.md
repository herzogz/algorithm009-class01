# 深度优先搜索和广度优先搜索

## 深度优先搜索、广度优先搜索的实现和特性

**遍历搜索**

我们将搜索化简，收缩到树或者图的情况下来进行搜索。

* 每个节点都要访问一次

* 仅访问一次（不做过多无用访问，不然效率低下）

* 对于节点的访问顺序不限

  * 深度优先：DFS

  * 广度优先：BFS

  * 自定义优先级（依据现实中的场景**优先级优先**：启发式搜索（深度学习，推荐算法））

## 深度优先

![深度优先搜索- 维基百科，自由的百科全书](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Depth-first-tree.svg/1200px-Depth-first-tree.svg.png)

### 示例代码

**DFS**

```python
def dfs(node):
  if node in visited:
    # already visited
    return
  visited.add(node)# 访问节点加到已访问节点中去
  
  #process current node
  # ...# logic here
  dfs(node.left)
  dfs(node.right)
```

**DFS多叉树情况**

```python
visited = set()
def dfs(node,visited):
  if node in visited: 
    # already visited
    return

  visited.add(node)
  # process current node here
  ...
  for next_node in node.children():
    if not next_node in visited:# 加入访问节点判断
      dfs(next_node, visited)
```

**在循环尚未结束时就会进入到下一层中**

**图的深度优先遍历顺序**

![img](https://github.com/wangkuiwu/datastructs_and_algorithm/blob/master/pictures/graph/iterator/04.jpg?raw=true)

### DFS-非递归写法

手动维护一个栈，来模拟递归

```python
def DFS(self,tree):
  if tree.root is None:
    return []
  visited,stack= [],[tree.root]
  
  while stack:
    node = stack.pop()
    visited.add(node)
    
    process(node)
    nodes = generate_related_nodes(node)# 注意顺序处理
    stack.push(nodes)
    
    # other processing work
    ...
```

## 广度优先

**不再使用递归或栈而使用队列，**一层一层向下扩散，想象成水滴，向下扩散，

现实中如地震，水波纹扩散等等。

![广度优先搜索- 维基百科，自由的百科全书](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Breadth-first_tree.svg/1200px-Breadth-first_tree.svg.png)



### 示例代码

```python
def BFS(graph,start,end):
  queue = []
  queue.append([start])
  visited.add(start)
  
  while queue:
    node = queue.pop() # 先入先出
    visited.add(node)
    
    process(node)
    nodes = generate_related_nodes(node)
    queue.push(nodes)
    
   # other processing work
  ...
```



## 实战题目解析



# 贪心算法 Greedy

>  贪心算法是一种每一步选择中都采取当前状态下最好或最优的选择，从而希望导致结果时**全局最好或最优的算法**。



* 贪心算法：对每个子问题的解决方案都做出选择，不能回退

  * 对于工程和生活的问题，贪心算法一般不能解决。可以解决一些最优化问题，如：求图中的最小生成树，求哈夫曼编码等。

  * 可以用作辅助算法，比较高效，直接解决一些要求结果不是很精确的问题。

* 动态规划：**会保存以前的运算结果**，并根据以前的结果对当前进行选择，有回退功能（最有判断+回溯）

## 贪心算法应用场景

* 问题能够分解成子问题来解决，子问题的最优解能递推到最终问题的最优解。这种子问题最优解称为**最优子结构**。
* 贪心法解题难点在于如何证明可以使用贪心算法
  * 有时可以直接使用
  * 有时需要将问题进行一定的转化
  * 从前往后，或从后往前

## 实战题目

### 445 分发饼干

让饼干最大可能被利用

* 将胃口数组升序排列，将饼干尺寸组也按升序排列

* 依次最小胃口，用大于等于其胃口最小饼干去满足，以此类推（从小到大匹配两个数组）

  * 满足胃口，数组分别+1
  * 不满足胃口，查看下一个小饼干，饼干组+1

  **时间复杂度**：$O(N)$

  

  ###  122 买卖股票最佳时机 Ⅱ

贪心的解法：

* 只要后一天比前一天大，那么就在前一天买，在后一天抛
* 只需遍历一次

**时间复杂度**：$O(N)$



### 55 跳跃游戏

> 可行：[2,3,1,1,4]
>
> 不可行：[3,2,1,0,4]，总会跳到索引3的位置

**数组方法**：

* 取一个数组放中间结果，可以到达的位置设为True，看是否可以到达最后
* 两重循环，时间复杂度：$O(N^2)$

**贪心方法**：

* **从后往前进行贪心** ，首先设置最后到达的索引初始值的下标为最后一个下标
* 从后往前循环，如果当前的i可以跳到最后，就把i更新到最后到达索引
* 查看最后到达索引是否为0，为0则表示第一个坐标的位置也可以跳到最后
* 时间复杂度：$O(N)$



# 二分查找

## 二分查找的前提

1. 目标函数单调性（递增或递减）
2. 存在上下界（bounded）
3. 能够通过索引访问（index accessible），单链表不行（需改造成跳表）。

## 代码模版

```python
## 升序排列的数组
left,right = 0,len(array)-1
while left <= right:
  mid = (left+right)/2
  if array[mid] == target:
    break or return result
  elif array[mid]<target:
    left = mid + 1 # 找右边
  else:
    right = mid - 1 # 找左边
```

 **这里边界为interger，为实数的情况下，直接left=mid**



## 实战题目

### 69 x的平方根

> x为非负整数，返回类型时整数，只保留整数部分

**二分查找**

* $y=x^2$，(x>0)：抛物线在y轴右侧单调递增，有上限界(0，x) ，满足二分查找的条件
* x的平方根肯定落在(1,x)之间，所以下界为1，上界为x
* 为了防止越界，做一个技术处理:mid = left + (right - left)/2

**牛顿迭代法**

[牛顿迭代法](https://www.beyond3d.com/content/articles/8/)

以直线代替曲线，用一阶泰勒展示（即当前点的切线）代替原曲线，求直线与x轴的交点，重复这个过程直到收敛。

```python 
def mySqrt(self,x):
  if x < 0:
    raise Exception('不能输入负数')
  if x == 0:
    return 0
  cur = 1
  while True:
    pre = cur
    cur = (cur + x / cur) / 2 # 无限逼近平方根
    if abs(cur - pre ) < 1e-6: # 当其和前一个数差别非常小的时候，说明基本已经得到所求平方根
      return int(cur)   
```

```python
def mySqrt(self,x):
  r = x
  while r*r > x:
    r = (r + x/r) / 2
    return r
```

## 搜索旋转排序数组

> 半有序数组：升序排序的数组在某点截断，进行了平移

1. 暴力：还原成升序（logn：使用二分查找找到被截断的位置），再二分查找

2. 二分查找
   * 筛选条件：
     * 首先看左边界和中间的值，如果单调递增，则判断target在哪边，不再是淡出判断mid，而是判断mid和左边界之间的关系
     * **如果不是单调递增，旋转为在（0，mid）之间，那么mid右边肯定是单调递增的**





