# 泛型递归，树的递归

> 树的面试题解法一般都是递归

1. 节点的定义
2. 重复性（自相似性）

## 递归 Recursion

* 递归-循环

* 通过函数体来进行循环

### 盗梦空间

* 向下进入到不同梦境中，向上又回到原来一层
* 声音同步回到上一层（同步关系用参数来进行函数不同层之间的传递变量） 
* 每一层的环境和周围的人都是一份拷贝，主角等几个人（==类似于函数的参数，同时还会有一些全局变量==）穿越不同层级的梦境（发生和携带变化）

### 求阶乘

计算n!
$$
n!=1\times2\times3\times...\times.n
$$


```python
def Facotrial(n):
  if n<=1:
    return 1
  return n * Factorial(n-1)
```

递归的运行方式调用一个**递归栈**，像一种剥洋葱的形式。

### 递归代码模版

* 递归终止条件
* 处理当前层逻辑
* 下探下一层
* 清理当前层（if needed）

#### python

```python
def recursion(level,param1,param2,...):
  # recursion terminator 
  if level>MAX_LEVEL:
    process_result
    return
  
  # process logic in current level
  process(level,data...)
  
  # drill down
  self.recursion(level+1,p1,...)
  
  # reverse the current level status if needed
```

#### java

```java
// Java
public void recur(int level, int param) { 
  // terminator 
  if (level > MAX_LEVEL) { 
    // process result 
    return; 
  }
  // process current logic 
  process(level, param); 
  // drill down 
  recur( level: level + 1, newParam); 
  // restore current status 
 
}
```



### 要点

1. 不要人肉递归
2. 找到最近最简方法，将其拆解成可重复解决的问题（**重复子问题**）
3.  数学归纳法思维（n=1,n=2成立，n成立时，n+1也成立）

---

# 分治，回溯的实现和特性

> 分治和回溯时递归的一个细分类，是一种特殊的递归
>
> (1) 找出简单的基线条件;
>
> (2) 确定如何缩小问题的规模，使其符合基线条件。
>
> 在平时日常生活中，分治思想也是随处可见的。例如：当我们打牌时，在进行洗牌时，若牌的数目较多，一个人洗不过来，则会将牌进行分堆，单独洗一小堆牌是相对容易的，每一堆牌都洗完之后再放到一起，则完成洗牌过程。

## 分治 divide and conquer，D&C

![浅谈什么是分治算法](http://www.cxyxiaowu.com/wp-content/uploads/2019/10/1571057768-709f1270922d265.jpeg)

本质上是寻找重复性，问题分解成多个子问题 ，最后组合子问题的结果

```python
def divide_conquer(problem,param1,param1,param2,...):
  # recursion terminator
  if problem is None:
    print_result
    return
  # prepare data
  data = prepare_data(problem)
  subproblems = split_problem(problem,data)
  #conquer subproblems
  subresult1 = self.divide_conquer(subproblems[0],p1,...)
  subresult2 = self.divide_conquer(subproblems[1],p1,...)
  subresult3 = self.divide_conquer(subproblems[2],p1...)
  ...
  # process and generate the final result
  result = process_result(subresult1,subresult2,subresult3,...)
```

**分治的特点在于最后需要把子问题结果组装起来返回**，同公司运作等现实场景类似。

### 归并排序

在常见排序如归并，快排中都采用了分治的思想

* 归并排序时一种自下而上的排序方法，先递归，再合并
* merge函数将已排好序的有序数组两两合并，==需要开辟新的存储空间消耗内存，空间复杂度为O(n)==
* 时间复杂度为O(nlogn)

```python
def merge(li, low, mid, high):
    i = low
    j = mid + 1
    l_tmp = []
    while i <= mid and j <= high:  # 只要左右两边都有数
        if li[i] < li[j]:
            l_tmp.append(li[i])
            i += 1
        else:
            l_tmp.append(li[j])  # 消耗内存
            j += 1
        # while 结束时， 可定有一部分没有数了
    while i <= mid:
        l_tmp.append(li[i])
        i += 1
    while j <= high:
        l_tmp.append(li[j])
        j += 1
        # copy回原数组
    li[low:high + 1] = l_tmp  # 包头不包尾



def merge_sort(li, low, high):
    if low < high:  # 保证列表有两个数
        mid = (low + high) // 2
        merge_sort(li, low, mid)
        merge_sort(li, mid + 1, high)
        merge(li, low, mid, high)


```





## 回溯 Backtracking

> 对于某些计算问题而言，回溯法是一种可以找出所有（或一部分）解的一般性算法，尤其适用于约束满足问题（在解决约束满足问题时，我们逐步构造更多的候选解，并且在确定某一部分候选解不可能补全成正确解之后放弃继续搜索这个部分候选解本身及其可以拓展出的子候选解，转而测试其他的部分候选解）。
>
> 回溯法采用==试错==的思想，它尝试分步的去解决一个问题。在分步解决问题的过程中，当它通过尝试发现现有的分步答案不能得到有效的正确的解答的时候，它将==取消==上一步甚至是上几步的计算，再通过其它的可能的分步解答再次尝试寻找问题的答案。回溯法通常用最简单的递归方法来实现，在反复重复上述的步骤后可能出现两种情况：
>
> 1. 找到一个可能存在的正确的答案
> 2. 在尝试了所有可能的分步方法后宣告该问题没有答案
> 3. 在最坏的情况下，回溯法会导致一次==复杂度为指数时间的计算==

### 典型应用

* 八皇后（八皇后问题是在标准国际象棋棋盘中寻找八个皇后的所有分布，使得没有一个皇后能攻击到另外一个。）
* 数独



