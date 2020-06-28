# 动态规划

> Dynamic Programming 动态递推

* 把复杂问题拆分为简单子问题（用一种递归的方式）
* 分治 + 最优子结构 Optimal substructure
  * 中间每一步不需要保存所有状态，只需要存最优的状态

## key point

* **动态规划**和**递归**或者**分治**没有根本上的区别（关键在于有无最优子结构）
* 共性：找到重复子问题
* 差异性：最优子结构，中途可以比较淘汰次优解（降低时间复杂度---》多项式级别）

## 解题思路

1. 找到状态和选择 
2. 明确 dp 数组/函数 的定义 
3. 寻找状态之间的关系。

****

# 实战例题

## 斐波拉契数列

* 增加一个缓存，记忆化搜索
* 自顶向下
* 时间复杂度为$O(N)$

```java
int fib (int n, int[] memo){
	if (n <= 1){
		return n;
	}
  if (memo[n] == 0){
    memo[n] - fib (n - 1) + fib (n - 2);
  }
  return memo [n];
}
```

---

* 自底向上 递推

* F[n] = F[n-1] + F[n-2]

* a[0] = 0,a[1] = 1

  for (int i =2 ; i <= n ; ++ i){

  ​	a[i] = a[i-1] + a[i-2]

  }

* a[n]
* 0,1,1,2,3,5,8,13

---

## 62 路径计数

绕过障碍物，有多少种路径

![截屏2020-06-22上午11.14.02](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-06-22上午11.14.02.png)

* pahts(start,end) = 
  * pahts(A,end) + paths(B,end)
    * paths(D,end)+paths(C,end)
    * paths(C,end)+paths(E,end)

* 状态转移方程（DP）![截屏2020-06-22上午11.26.05](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-06-22上午11.26.05.png)
* 

```
opt[i,j] = opt[i+1,j]+opt[i,j+1]

if a[i,j] = '空'：
	opt[i,j] = opt[i+1,j] + opt[i,j+1]	
else: # 障碍物
	opt[i,j] = 0
```

### 动态规划关键点

* 最优子结构 optp[n] = best_of(opt[n-1],opt[n-2],...)
* ==存储中间状态：opt[i]==
* 递推公式（状态转移方程）
  * Fib:`opt[i]=opt[n-1]+opt[n-2]`
  * 二维路径：`opt[i,j] = opt[i+1][j]+opt[i][j+1](判断a[i,j]是否空地)`

---

## 64 最小路径和

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        for m in range(len(grid)):
            for n in range(len(grid[0])):
                if m == n == 0:
                    continue
                elif m == 0 :
                    grid[m][n] += grid[m][n-1]
                elif n == 0:
                    grid[m][n] += grid[m-1][n]
                else:
                    grid[m][n] += min(grid[m-1][n],grid[m][n-1])
        return grid[-1][-1]
```



## 1143 最长公共子序列

## 198 打家劫舍

思路：

* a[i]: 0..i 能偷到max value：a[n-1]
* 增加一个维度，来表示房子是否被偷的状态
* `a[i][0,1]`:1： i个房子偷，0：i个房子不偷
* `a[i][0] = Max(a[i-1][0],a[i-1][1])`
* `a[i][1] = Max(a[i-1][0],0)+nums[i]` ，可以偷第i个房子的情况ß

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        n = len(nums)
        dp = [0 for _ in range(n+1)]
        dp[0] = 0
        dp[1]  = nums[0]
        for k in range(2,n+1):
            dp[k] = max(dp[k-1], dp[k-2]+nums[k-1])
        return dp[n]
```

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        pre , cur = 0 , 0
        for i in nums:
            pre,cur = cur,max(cur, pre + i) # 前一个偷，或者前前一个偷加上现在一个的最大值
        return cur

```

## 213 打家劫舍 2

拓展版，变成两个单列，一个包含第一家，一个不包含第一家，最后比较大小

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        def rob_house(nums):
            pre,cur = 0,0
            for i in nums:
                pre, cur = cur , max(cur,pre+i)
            return cur
        return max(rob_house(nums[1:]),rob_house(nums[:n-1])) if n != 1 else nums[0] # 比较不同的两段
```

## 91 解码方法

* 类似于爬楼梯问题

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        pre, cur = 1, int(s[0] != '0')
        for i in range(1, len(s)):
            pre, cur = cur, pre * (9 < int(s[i-1:i+1]) <= 26) + cur * (int(s[i]) > 0)
        return cur
```

## 221 最大正方形

`dp[i][j] = min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]) + 1`

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return 0
        # 最大边长初始化
        maxside = 0
        row = len(matrix)
        column = len(matrix[0])
        dp = [[0] * column for _ in range(row)]
        for i in range(row):
            for j in range(column):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0: # 如果为0行和0列时，dp值最大为1
                        dp[i][j] = 1
                    # 不在第0行和第0列的情况
                    # 当前作为正方形右下角，其最大边长为左，上，左上放最小值+1
                    else:
                        dp[i][j] = min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]) + 1
                    maxside = max(maxside,dp[i][j])
        return maxside * maxside
                    
        
```

## 621 任务调度器

```python
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:

# 最短时间需要把各种不同的任务排在一起执行
# 先排列好任务数最多的任务，再把其他少于最多任务的数的任务循环插入再间隔时间中
        count = [0] * 26
        for ch in tasks:
            count[ord(ch) - ord('A')] += 1

        count_max = max(count)
        total = (count_max - 1) * (n + 1)

        for _ in count:
            if _ == count_max:
                total += 1

        return max(total, len(tasks))

```

