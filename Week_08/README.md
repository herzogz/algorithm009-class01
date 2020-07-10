# 位运算

为什么需要位运算？

* 机器里的数字表示方式和存储格式就是**二进制**
* 十进制与二进制之间的相互转换

## 位运算符

|   含义   | 运算符 | 示例                   |
| :------: | :----: | ---------------------- |
|   左移   |  $<<$  | 0011=>0110             |
|   右移   | $ >>$  | 0110=>0011             |
|  按位或  |  $|$   | 0011 \| 1011  --->1011 |
|  按位与  |   &    | 0011 & 1011 --->0011   |
| 按位取反 |   ~    | ~ 0011 ---> 1100       |
| 按位异或 |   ^    | 0011 ^ 1011 ---> 1000  |

### XOR - 异或

相同为0，不同为1，也可以用“不进位加法”来理解。

* x ^ 0 = x
* x ^ 1s = ~ x    // 1s = ~ 0
* x ^ (~ x) = 1s
* x ^ x = 0
* c = a ^ b ----> a ^ c  = b , b ^ c = a //交换两个数
* a ^ b ^ c = a ^ (b ^ c) = (a ^ b) ^ c  //结合法

## 指定位置的位运算

| 需求                         |     操作      |
| :--------------------------- | :-----------: |
| 将x最右边的n位清零           | x & (~ 0 <<n) |
| 获取x的第n位值（0或者1）     |   (x>>n)&1    |
| 获取x的第n位的冥值           |   x&(1<<n)    |
| 仅将第n位置为1               |   x\|(1<<n)   |
| 仅将第n位置为0               |   x&(1<<n)    |
| 将x最高位置为第n位（含）清零 | x&((1<<n)-1)  |

## 实战位运算要点

位运算比模快不少

* 判断奇偶

  * `x%2==1  (if x%2)------>(x&1)==1`
  * `x%2==0  (if not x%2)------>(x&1)==0`

* `x>>1 ------> x/2`

  * `mid = (left+right)/2  -----> mid = (left+right)>>1`

* `x = x&(x-1)`：清零最低位的1

* `x&-x`:  得到最低位的1，负数以原码的补码形式表达。

* `x&~x`: 0

  

## 实战题目

### 191 位1的个数

移动32次

* 循环 for loop 0---->32
* 先%2 判断最后一位是否为1，/2再把最后一位去掉
* &1 判断是否最后一位是否为1，x = x>>1右移一位 

* while(x>0){count++;x = x&(x-1)}，x每次清掉最低位的1（不在是32次，多少个1循环多少次）

**位运算**

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n>0:
            n &= n-1
            count += 1
        return count
```

### 231 2的冥

2的冥次方表示整数的二进制有且仅有一个1

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return (n > 0) and (n & (n - 1) == 0)
```

### 190 颠倒二进制位

* int --->'010101' string ----> reverse
* int ---> 位运算

```python
class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for _ in range(32):
            res = (res<<1) + (n&1) # 最后一位挪到前面去
            n >>= 1 再把最后一位去掉
        return res

```



```python
class Solution:
    def reverseBits(self, n: int) -> int:
      ans,mask = 0, 1
      for i in range(32):
        if n & mask :
          ans |= 1 << (31-i) # 置1操作
        mask <<= 1
      return ans
      
```

### 52 n皇后2

使用位运算，取代判重的数组

```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        if n < 1:
            return 0
        self.count = 0
        self.dfs(n,0,0,0,0)
        return self.count
    def dfs(self, n, row, col, pie ,na):
        # terminator
        if row >= n:
            self.count += 1
            return 
        position = (~(col | pie | na)) & ((1 << n) - 1) # 将最高位为n位清零，整理出皇后可以放的位置
        while position: # 还有位置可以放皇后的时候
            p = position & -position # 取最低位的1
            position &= position - 1 # 清除最低位的1， 表示把皇后放过去
            self.dfs(n, row + 1, col | p, (pie | p) << 1, (na | p) >> 1)
```

### 338 比特位计数

* dp问题: dp[i] = dp[i>>1] + (i&1)
  *  i>>1代表前一个二进制位的次数
  *  i&1代表i的末尾是否为1

```python
def countBits(num):
    dp = [0]
    for i in range(1, num + 1):
        dp.append(dp[i>>1] + (i&1))
    
    return dp

print(countBits(5))

```

**奇偶规律**

* 奇数的二进制中一的位数等于前面的偶数的二进制中1的个数加1,前面偶数的最后一位的0变成1
* 偶数的二进制中一的位数等于自身除以2的数的二进制中1的个数,偶数除以2相当于右移一位,也就是将最后面的0移出去,不影响1的个数

```python
class Solution:
    def countBits(self, num: int) -> List[int]:
        dp=[0]*(num+1)
        for i in range(1,num+1):
            if i & 1:
                dp[i]=dp[i-1]+1
            else:
                dp[i]=dp[i//2]
        return dp
```

**位1计数升级**

```python
class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        result = []
        for i in range(num+1):
            temp = 0
            while i:
                i &= i-1
                temp += 1
            result.append(temp)

        return result

```

---

# 布隆过滤器、LRU Cache

## Bloom Filter

一个很长的**二进制**向量和一系列**随机映射函数**。布隆过滤器可以用于检索一个元素是否在一个集合中（不能存储其他额外的信息）。

* 优点是空间效率和查询时间都远超一般的算法
* 缺点是有一定的误识别率和删除困难（模糊查询）

![布隆过滤器示意图](https://raw.githubusercontent.com/lonelyswan/lonelyswan.github.io/master/images/649px-Bloom_filter.svg.png "布隆过滤器示意图")

**只要有一个二进制位为0，则查找的元素不在布隆过滤器中，但是所有二进制位皆为1时，查找的元素不一定（仅可能）在布隆过滤器中（二进制位有重合）。**

布隆过滤器只是在机器前面的快速查询的缓存，真正要确定元素是否存在，需要访问机器里的一个完整的存储数据结构（一般来说是数据库）。

![截屏2020-07-07上午10.32.28](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-07-07上午10.32.28.png)

### 应用

**不是的话肯定不是，是的话在进一步具体的查询**

1. 比特币网络
2. 分布式系统（Map-Reduce）-------Hadoop、search engine
3. Redis 缓存
4. 垃圾邮件、评论等的过滤

### python 实现

```python
from bitarray import bitarray
import mmh3

calss BloomFilter(object):
  def __init__(self,size,hash_num):
    self.size = size
    self.hash_num = hash_num
    self.bit_array = bitarray(size)
    self.bit_array.setall(0)
    
  def add(self,s):
    for seed in raneg(self.hash_num):
      result = mmh3.hash(s, seed) % self.size
      self.bit_array[result] = 1 # 将索引位置置为1
     
  def lookup(self, s):
    for seed in range(self.hash_num):
      result = mmh3.hash(s, seed) % self.size
      if self.bit_array[result] == 0: # 只要有一位包含0
        return 'Nope'
      return 'Probably'
    
    
```

### 参考

[使用BloomFilter布隆过滤器解决缓存击穿、垃圾邮件识别、集合判重](https://blog.csdn.net/tianyaleixiaowu/article/details/74721877)

[python 高性能布隆过滤器](https://github.com/jhgg/pybloof)

---

## LRU Cache

### Cache 缓存

1. 记忆（@ LRU Cache）
2. 两个要素：大小，替换策略
   1. LFU
   2. LRU
3. Hash Table + Double LinkedList
4. 复杂度

   * 查询：$O(1)$
   * 修改：$O(1)$

### 工作机制

**最近被使用的元素永远放在最前面**

![截屏2020-07-07上午11.04.56](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-07-07上午11.04.56.png)

### python 实现

```python
class LRUCache(object): 
	def __init__(self, capacity): 
		self.dic = collections.OrderedDict() 
		self.remain = capacity
	def get(self, key): 
		if key not in self.dic: 
			return -1 
		v = self.dic.pop(key) 
		self.dic[key] = v   # key as the newest one 
		return v 
	def put(self, key, value): 
		if key in self.dic: 
			self.dic.pop(key) # 先弹出
		else: 
			if self.remain > 0: 
				self.remain -= 1 
			else:   # self.dic is full
				self.dic.popitem(last=False) 
		self.dic[key] = value # 放到最前面

```

---

# 排序算法

1. 比较类排序

   通过比较来决定元素间的相对次序，由于其时间复杂度不能突破$O(nlogn)$，因此也称为非线性时间比较类排序。时间复杂度O(nlogn) ~ O(n^2)，主要有：**冒泡排序，选择排序，插入排序，归并排序，堆排序，快速排序等。**

2. 非比较类排序

   不通过比较来决定元素间的相对次序(**一般对于整型的元素**），它可以突破基于比较排序的时间下界，以现行时间运行，因此可以称为线性时间非比较类排序。时间复杂度可以达到O(n)，主要有：**计数排序，基数排序，桶排序等。**

![截屏2020-07-08下午12.47.06](/Users/echotrees/Desktop/截屏2020-07-08下午12.47.06.png)

参考：[线性时间排序算法](https://www.cnblogs.com/gaochundong/p/sorting_in_linear_time.html)

3. [排序的时间复杂度、空间复杂度和稳定性总结](https://www.bigocheatsheet.com/)

![截屏2020-07-08下午12.47.43](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-07-08下午12.47.43.png)

4. [十大经典排序算法（动图演示）](https://www.cnblogs.com/onepixel/p/7674659.html)

## 初级排序和高级排序的实现和特性

### 初级排序 - $O(n^2)$

1. 选择排序（Selection Sort）

   每次找最小值，然后放到待排序数组的起始位置

   ```python
   def select_sort(li):
       for i in range(len(li) - 1):
           min_index = i  # 无序区第一个数下标
           for j in range(i + 1, len(li)):
               while li[j] < li[min_index]:
                   min_index = j  # 新的最小数的下标，无序区最小数的下标
           li[i], li[min_index] = li[min_index], li[i]  # 和无序区第一个数交换
       print(li)
   
   a = [3,4,5,2,12,3,4,6,7,8,9,9,1,1,0,2,3]
   select_sort(a)
   ```

   

2. 插入排序（Insertion Sort）

   从前到后逐步构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入

   **类似于打扑克牌，摸牌后，把最小的牌按顺序依次插入到最前面**

   ```python
   #!/usr/bin/env python3
   # -*- encoding:utf-8 -*-
   def insert_sort(li):
       for i in range(1, len(li)): # 准备排序的牌
           tmp = li[i] # 准备排序的元素
           j = i - 1 # 从后向前扫描的下标起点
           while j >= 0 and li[j] > tmp:
               li[j+1] = li[j] # 如果大于准备插入的数，则向后挪
               j -= 1
           li[j + 1] = tmp
       print(li)
   
   a = [4,4,6,7,5,2,1,8,10]
   insert_sort(a)
   ```

3. 冒泡排序（Bubble Sort）

   嵌套循环，每次查看相邻元素，如果逆序，则交换

   一共只需要走n-1趟就行，最后一次自动排好序

   **需要注意的是，排序过程中可能已经排好序了，所以避免多余的扫描，需要在中间设置一个发生交换的信息，如果没有交换发生，则不需要再扫描下去了**

   ```python
   def bubble_sort(li):
       for i in range(len(li) - 1):
           exchange = False # 交换信号
           for j in range(len(li) - i - 1):
             # 升序，降序的话把小的数翻上去 "<"
               if li[j] > li[j + 1]:
                   li[j], li[j + 1] = li[j + 1], li[j]
                   exchange = True # 发生了交换
           if not exchange:
               print(li)
               return
       print(li)
   
   a = [1,2,3,4,4,5,3,2]
   bubble_sort(a)
   
   ```

   

### 高级排序 - $O(N*logN)$

#### 快速排序（Quick Sort）

**自上而下，先分区，再递归**

数组取基准点pivot，将小元素放pivot左边，大元素放右侧门然后一次对右边和左边的字数组继续快排，以达到整个序列有序

![AlgorithmHub - Quick-Sort](https://i2.wp.com/www.techiedelight.com/wp-content/uploads/Quicksort.png?w=1100http://)

```python
def quick_sort(li, left, right):
    if left < right:
        mid = partition(li, left, right) # 先分区，再递归
        quick_sort(li, left, mid - 1)
        quick_sort(li, mid + 1, right)

def partition(li, left, right):
    """区分函数"""
    tmp = li[left]
    while left < right:
        while left < right and li[right]>= tmp: # 从右面找比tmp小的数
            right -=1 #往左走一步,如果碰上了tmp，则退出
        li[left] = li[right] # 如果right= left 则自己赋值自己
        while left < right and li[left]<= tmp:
            left += 1
        li[right] = li[left] # 把左边的值写到右边的空位
    li[left] = tmp # 最后碰到一起后，把原来值写到最后的空位
    return left

li = [1, 2, 3, 6, 3, 2, 1, 10, 7, 8]
quick_sort(li, 0, len(li) - 1)
print(li)

```



#### 归并排序（Merge Sort）

![归并排序改进归并_anlian523的博客-CSDN博客_改进的归并算法排序,链接 ...](https://img-blog.csdnimg.cn/20181113152916219.png)

**自下而上，先递归，再归并**

* 把长度为n的输入序列分成两个长度为n/2的子序列
* 对这两个字序列分别采用归并排序
* 将两个排序好的子序列合并成一个最终的排序序列

**步骤**

1. 分解，将列表越分越小，直至分为一个元素
2. 终止条件，一个元素是有序的
3. 合并，将各个有序列表归并，列表再组合成原始最长的长度

```python
import random

def merge_sort(li, low, high):
  # terminator
    if low < high:  # 保证列表有两个数
        mid = low + （(high - low) >> 2）  # 复杂度折半循环 O(logn)
        merge_sort(li, low, mid) # 先递归，再合并
        merge_sort(li, mid + 1, high)
        merge(li, low, mid, high) # 每层相当于遍历完整数组一遍，每层的时间复杂度为O(n)
# 合并函数       
def merge(li, low, mid, high):
    i = low
    j = mid + 1
    l_tmp = []
    while i <= mid and j <= high:  # 只要左右两边都有数
        if li[i] < li[j]: # 比较两边
            l_tmp.append(li[i])
            i += 1
        else:
            l_tmp.append(li[j])  # 消耗内存
            j += 1
        # while 结束时， 肯定有一部分没有数了
    while i <= mid: # 如果左边还有剩余的数
        l_tmp.append(li[i]) # 按顺序添加剩余没有选出来的值
        i += 1
    while j <= high: # 如果有边还有剩余的数
        l_tmp.append(li[j])
        j += 1
        # copy回原数组
    li[low:high + 1] = l_tmp  # 包头不包尾
    #li[start:end+1] = l_tmp + nums[l:mid+1] + nums[r:end+1]

li1 = list(range(10))
random.shuffle(li1)
print(li1)
merge_sort(li1, 0, len(li1) - 1)
print(li1)
```



#### 堆排序 （Heap Sort）

[python堆排序内置模块heapq](https://blog.csdn.net/qq_32617703/article/details/101707913)

* 堆排序 -- 插入$O(logN)$，取最大/最小值$O(1)$
  * 数组元素依次建立小顶堆
  * 依次去堆顶元素，并删除

## 特殊排序 - $O(n)$

1. 计数排序（Counting Sort）

   计数排序要求输入的数据必须是有确定范围的整数，将输入的数据值转化为键存储在**额外开辟的数组空间中**，然后依次把计数大于1的填充回原数组

2. 桶排序

假设输入数据服从均匀分布，将数据分到有限数量的桶里，每个桶再分别排序（有可能再使用别的排序算法或者以递归方式继续使用同排序）

3. 基数排序

   基数排序是按照低位先排序，然后收集，再按照高位排序，然后再收集；依次类推，直到最高位。有时候有些属性是有优先级顺序的，先按低优先级排序，再按高优先级排序

## 实战题目

### 1122 数组的相对排序

**使用计数排序**

```python
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        bins = [0 for _ in range(1001)] # 下标为值，值为频次
        res = []
        for i in arr1:
            bins[i] += 1
        for i in arr2: # 处理在arr2中出现过的,并按照arr2的顺序取出
            res += [i] * bins[i] # 值 * 频次
            bins[i] = 0
        for i in range(len(bins)): # 处理剩余的不在arr2中出现过的
            res += [i] * bins[i] # 其他所有位皆为0，只有还剩下没有出现过的元素按顺序添加
        
        return res
```

### 56 合并区间

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) == 0:
            return []
        res = []
        intervals.sort(key=lambda x: x[0])  # 先按区间左边界值由小到大排序
        for inter in intervals:
            if len(res) == 0 or res[-1][1] < inter[0]:  # 如果结果集最后一个元素的右边界比新加入区间的左边界小，直接加入结果集
                res.append(inter)
            else:  # 说明新加入的和结果集最后一个区间有重合，更新区间右边界即可
                res[-1][1] = max(res[-1][1], inter[1])
        return res
```

参考自：Lotus panda

### 493 翻转对（逆序对）

* 暴力法 O(N^2)
* merge-sort
* 树状数组（二进制方法罗列下标）

**归并**

在归并之前循环变量统计一下两边有序数组构成的翻转对的数量;

**时间复杂度**o(nlgn)

1. 分治归并的时候要做两件事情：计算跨两部分的重要翻转对数；对两部分进行归并排序。

```python
class Solution:
    # 归并排序 过程中计数， 时间复杂度 O(NlogN)
    def merge(self, nums, start, mid, end):
        l, r = start, mid+1
        res = []
        while l <= mid and r <= end:
            if nums[l] >= nums[r]:
                res.append(nums[r])
                r += 1
            else:
                res.append(nums[l])
                l += 1
        nums[start:end+1] = res + nums[l:mid+1] + nums[r:end+1]

    def mergesort_and_count(self, nums, start, end):
        if start >= end:
            return 0
        mid = start + ((end - start) >> 2) # 使用位运算速度更快
        count = self.mergesort_and_count(nums, start, mid) + \
                self.mergesort_and_count(nums, mid+1, end)
        j = mid + 1
        for i in range(start, mid+1):
            while j <= end and nums[i] > 2 * nums[j]:
                j += 1
            count += j - (mid + 1)
        self.merge(nums, start, mid, end)
        return count

    def reversePairs(self, nums: List[int]) -> int:
        return self.mergesort_and_count(nums, 0, len(nums)-1)

```

















