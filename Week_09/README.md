# 高级动态规划

## 复习

* 人肉递归低效、很累
* 找到最近最简方法，将其拆解成可重复解决的问题（最大公约数）
* 数学归纳法思想
* **本质：寻找重复性-----------》计算机指令集**

## 动态规划、状态转移方程

> 动态规划和递归分治没有本质上的的区别（关键看有无最优子结构）
>
> 共性：找到重复的子问题
>
> 差异：最优子结构，中途可以淘汰次优解

* 复杂分体分解为简单的子问题
* 动态规划的本质：分治 + 最优子结构
* 顺推形式（自下往上）： 动态递推

```java
function DP():
	dp = [][] # 二维
	
	for i = 0 .. M{
		for j = 0 ..N{
		dp[i][j] = _Funciton(dp[i'][j']...)
      # 求最小值，累加或累减，或者有一层小的循环，从之前的k个状态里找出最值
		}
	}
	return dp[M][N];
```

### 爬楼梯

* 转换斐波拉契

* 硬币置换异曲同工



### 不同路径

`f(x,y) = f(x-1,y) + f(x,y-1)`

```python
def f(x,y):
	if x <= 0 or y <= 0: return 0
	if x == 1 and y == 1: return 1
	return f(x-1,y) + f(x,y - 1)
```

缓存方法，复杂度O(mn),O(mn)

```python
def f(x,y):
	if x <= 0 or y <= 0:return 0
  if x == 1 and y == 1:return 1
  if (x,y) not in mem:
    mem[(x,y)] = f(x - 1, y) + f(x, y - 1)
  return mem[(x,y)]
```

动态递推，复杂度，O(mn),O(mn)

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        f = [[0]*n for zong in range(m)]
        for i in range(m):
            f[i][0] = 1
        for j in range(n):
            f[0][j] = 1
        for i in range(1,m):
            for j in range(1,n):
                f[i][j] = f[i-1][j]+f[i][j-1]
        return f[-1][-1]
```

### 63 不同路径2

* 动态规划，先处理第一行和第一列，然后再处理其他的
* 状态方程为f(i,j)=f(i-1,j)+f(i,j-1)，有障碍物的话就是f(i,j)=0了。
* 另外测试用例还往入口放了个障碍物

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        if obstacleGrid[0][0] == 1:
            return 0
        dp = [[0 for _ in range(n)] for _ in range(m)]
        dp[0][0] = 1
        for i in range(1, m):
            if obstacleGrid[i][0] == 0:
                dp[i][0] = dp[i-1][0]
        for j in range(1, n):
            if obstacleGrid[0][j] == 0:
                dp[0][j] = dp[0][j-1]
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]

```





### 打家劫舍

状态定义

`dp[i] = max(dp[i-2]+nums[i], dp[i-1])`

加一维状态

`dp[i][0] = max(dp[i-1][0],dp[i-1][1])`

`dp[i][1] = dp[i-1][0]+ nums[i]`

### 最小路径和

状态定义

`dp[i][j] = min(dp[i-1][j], dp[i][j-1] + A[i][j])`

### 72 编辑距离

`dp[i][j] 代表 word1 到 i 位置转换成 word2 到 j 位置需要最少步数`

所以，

`当 word1[i] == word2[j]，dp[i][j] = dp[i-1][j-1]；`

`当 word1[i] != word2[j]，dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1`

`其中，dp[i-1][j-1] 表示替换操作，dp[i-1][j] 表示删除操作，dp[i][j-1] 表示插入操作。`



```python
#dp[i][j] // word1.substr(0,i)与word2.substr(0,j)之间的编辑距离
edit_dist(i,j) = edit_dist(i-1, j-1) if w1[i] == w2[j] #分治，两个字符相同，直接字符-1，不增加编辑次数

else // w1[i] != w2[j] # 要增加编辑次数
edit_dist(i, j) = 
MIN(edit_dist(i-1,j-1)+1,# 都减少说明替换
		eidt_dist(i-1, j)+1, # 删除其中一个
		edit_dist(i, j-1)+1)
```

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]

        # 状态初始化
        for i in range(n1 + 1):
            dp[i][0] = i
        for j in range(n2 + 1):
            dp[0][j] = j

        # 状态转移
        for i in range(1,n1 + 1):
            for j in range(1,n2 + 1):
                # 情况1，字符相等
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                # 情况2，字符不想等
                else:
                    dp[i][j] = min(dp[i][j-1],dp[i-1][j],dp[i-1][j-1]) + 1
        return dp[-1][-1]


```

#### 

---

# 字符串算法

* python和java的字符串是immutable，修改字符串实际上是新生成了一个符串
* C++的字符串是可变的

[不可变字符串](https://lemire.me/blog/2017/07/07/are-your-strings-immutable/)

## 基础问题

### 709 转换成小写字母

```python
# 'A' - 'Z' 对应的 ascii 是 65 - 90；
# 'a' - 'z' 对应的 ascii 是 97 - 122；
# 大小字母转换相差32，解题只要记住ord(),chr()函数即可
class Solution:
    def toLowerCase(self, str: str) -> str:
        s = []
        for i in str:
            if  65 <= ord(i) <= 90:
                s.append(chr(ord(i) + 32))
            else:
                s.append(i)
        return ''.join(s)
```

或者直接调函数str.lower()

### 58 最后一个单词的长度

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        if not len(s):
            return 0
        ss = s.rstrip() # 去掉最后一个空格
        for i in range(len(ss)-1,-1,-1): # 从后往前遍历
            if ss[i]==' ':
                return len(ss)-1-i
        return len(ss)
```

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        if not len(s):
            return 0
        return len(s.strip().split(' ')[-1])
```

### 771 宝石与石头

```python
class Solution:
    def numJewelsInStones(self, J: str, S: str) -> int:
        return sum([S.count(x) for x in J])
```

### 387 字符串中的第一个唯一字符

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        if not s:
            return -1
        counter = collections.Counter(s)
        for char in counter:
            if counter[char] == 1:
                return s.index(char)
        return -1
```

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        if not s:
            return -1
        hashmap = collections.defaultdict(int)
        for char in s:
            hashmap[char] += 1
        for char in hashmap:
            if hashmap[char] == 1:
                return s.index(char)
        return -1
```

### 8 字符串转换整数（atoi）

```python
class Solution:
    def myAtoi(self, str: str) -> int:
        ls = list(str.strip())
        if not ls : 
            return 0
        
        sign = -1 if ls[0] == '-' else 1
        if ls[0] in ['-','+'] : del ls[0]
        ret, i = 0, 0
        while i < len(ls) and ls[i].isdigit() :
            ret = ret*10 + ord(ls[i]) - ord('0')
            i += 1
        return max(-2**31, min(sign * ret,2**31-1))

```

### 14 最长公共前缀

1. 纯暴力
2. 行列遍历，输出公共前缀，两层循环
3. Trie字典树，放入字典树，然后看在第几层

### 344 反转字符串

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        start, end = 0, len(s) - 1
        while start < end:
            s[start],s[end] = s[end], s[start]
            start += 1
            end -= 1
```

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        s[:] = s[::-1]
        
```

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        i = len(s)-1
        tmp = ''
        while i >= 0: # 翻转全部字符串
            tmp += s[i]
            i -= 1
```



### 541 反转字符串2

每间隔k个反转k个字符。

利用Python切片其实不需要考虑“剩余字符”的长度这一问题

```python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        result=''
        for i in range(0,len(s),2*k):
            tmp=s[i:i+k] # k个数字反转
            tmp=tmp[::-1]+s[i+k:i+2*k] # 加上没有翻转的部分
            result += tmp
        return result

```

### 151 翻转字符串里的单词

**split()的时候，多个空格当成一个空格；split(' ')的时候，多个空格都要分割，每个空格分割出来空。**

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        ls = s.strip().split(' ')
        ls[:] = ls[::-1]
        return ' '.join([i for i in ls if i]) # 排除多个空格的可能性
```

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        ls = s.strip().split() # split()直接将多个空格看作一个空格
        ls[:] = ls[::-1]
        return ' '.join(ls) # 排除多个空格的可能性
```

**两次翻转**

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        i = len(s)-1
        tmp = ''
        while i >= 0: # 翻转全部字符串
            tmp += s[i]
            i -= 1
        # 翻转单词
        return ' '.join([i[::-1] for i in tmp.split()])
```

---

## 高级字符串算法

### 最长子串&子序列问题

#### 1143 最长公共子序列

* 状态转移方程：

  ```python
  if s1[i-1] == s2[j-1]:
  	dp[i][j] = dp[i-1][j-1] + 1
  else:
  dp[i][j] = max(dp[i-1][j], dp[i][k-1])
  ```

  ```python
  class Solution:
      def longestCommonSubsequence(self, text1: str, text2: str) -> int:
          m = len(text1)
          n = len(text2)
          dp = [[0] * (n + 1) for _ in range(m + 1)]
  
          for i in range(1, m + 1):
              for j in range(1, n + 1):
                  if text1[i-1] == text2[j-1]:
                      dp[i][j] = dp[i-1][j-1] + 1 # 说明已经找到了一个公共字符
                  else:
                      dp[i][j] = max(dp[i-1][j], dp[i][j-1]) # 没有找到的话，就删除其中一个字符串的字符
          return dp[m][n]
  ```

* 最长子串

  `dp[i][j]=dp[i-1][j-1] + 1(if s1[i-1]==s2[j-1]) else dp[i][j] = 0`

  因为不能删除中间的字符，所以如果不想等，则直接为0

#### 5 最长回文子串

* 枚举回文串的中心点，向外扩张，保证左右两边相同，要是不同，则停止扩张

  * 中心可以是一个元素，也可以是两个元素之间

  * ```python
    class Solution:
        def longestPalindrome(self, s: str) -> str:
            size = len(s)
            if size < 2:
                return s
    
            # 至少是 1
            max_len = 1
            res = s[0]
    
            for i in range(size):
                palindrome_odd, odd_len = self.__center_spread(s, size, i, i)
                palindrome_even, even_len = self.__center_spread(s, size, i, i + 1)
    
                # 当前找到的最长回文子串
                cur_max_sub = palindrome_odd if odd_len >= even_len else palindrome_even
                if len(cur_max_sub) > max_len:
                    max_len = len(cur_max_sub)
                    res = cur_max_sub
    
            return res
    
        def __center_spread(self, s, size, left, right):
            """
            left = right 的时候，此时回文中心是一个字符，回文串的长度是奇数
            right = left + 1 的时候，此时回文中心是一个空隙，回文串的长度是偶数
            """
            i = left
            j = right
    
            while i >= 0 and j < size and s[i] == s[j]:
                i -= 1
                j += 1
            return s[i + 1:j], j - i - 1 # 注意边界
    
    ```

    - 时间复杂度：O(N^{2})*O*(*N*2)，理由已经叙述。
    - 空间复杂度：O(1)*O*(1)，只使用到常数个临时变量，与字符串长度无关。

* 动态规划

  * `p(i,j) = (p(i+1,j-1)&&S[i]==S[j])` 首先首尾字符相同，且中间也是回文子串

  * 边界条件是：表达式 [i + 1, j - 1] 不构成区间，即长度严格小于 2，即 j - 1 - (i + 1) + 1 < 2 ，得 j - i < 3。

    ```
    class Solution:
        def longestPalindrome(self, s: str) -> str:
            size = len(s)
            if size < 2:
                return s
    
            dp = [[False for _ in range(size)] for _ in range(size)]
    
            max_len = 1
            start = 0
    
            for i in range(size):
                dp[i][i] = True
    
            for j in range(1, size):
                for i in range(0, j):
                    if s[i] == s[j]:
                        if j - i < 3:
                            dp[i][j] = True
                        else:
                            dp[i][j] = dp[i + 1][j - 1]
                    else:
                        dp[i][j] = False
    
                    if dp[i][j]:
                        cur_len = j - i + 1
                        if cur_len > max_len:
                            max_len = cur_len
                            start = i
            return s[start:start + max_len]
    
    
    ```

#### 10 正则表达式匹配

* 递归思想

  * ```python
    def isMatch(text,pattern):
    	if pattern is empty:
        return text is empty
      first_match = (text not empty) and pattern[0] == text[0]
      return first_match and isMatch(text[1:],pattern[1:])
    	
    ```

* 处理「.」

  ```python
  def isMatch(text,pattern):
  	if not pattern :
      return not text
    first_match = boll(text)  and pattern[0] in (text[0],'.')
    return first_match and isMatch(text[1:],pattern[1:])
  ```

* 处理「*」通配符

  ```python
  def isMatch(text,pattern):
  	if not pattern :
      return not text
    first_match = bool(text) and pattern[0] in {text[0],'.'}
    if len(pattern) >= 2 and pattern[1] == '*':
    # 通配符必须前面带一个字符
    # 星号当作0，略过两位
    	return isMatch(text,pattern[2:]) or \
    				first_match and isMatch(text[1:],pattern)
    else:
    return first_match and isMatch(text[1:],pattern[1:])
  ```

* 带备忘录的递归

  ```python
  def isMatch(text, pattern) -> bool:
      memo = dict() # 备忘录
      def dp(i, j):
          if (i, j) in memo: 
            return memo[(i, j)]
          if j == len(pattern): 
            return i == len(text)
  
          first = i < len(text) and pattern[j] in {text[i], '.'}
          # j<=len(pattern)-2,以保证pattern 长度大于等于2
          if j <= len(pattern) - 2 and pattern[j + 1] == '*':
              ans = dp(i, j + 2) or \
                      first and dp(i + 1, j)
          else:
              ans = first and dp(i + 1, j + 1)
              
          memo[(i, j)] = ans
          return ans
      
      return dp(0, 0)
  
  ```

  

* 动态规划

`dp[i][j]`

  

## 字符串匹配算法

有A，B两个字符串，问A在B中什么位置出现（反之亦然）。

1. 暴力法 - O(mn)

```python
def forceSearch(txt, pat):
  n, m = len(txt), len(pat)
  for i in range(n-m+1):
    for j in range(m):
      if txt[i+j] != pat[j]:
        break
    if j == m:
      return i
  return -1 

```

1. Rabin-Karp算法

   为了避免挨个字符对目标字符串和子串进行比较，我们可以尝试一次性判断两者是否相等。

   预先判断，通过哈希函数，酸楚子串的哈希值，用子串的哈希值来同目标字符串中的子串的哈希值进行比较，进而加速。

   * 假设子串长度M（pat），目标字符串的长度为N（txt）
   * 计算子串的hash值hash_pat
   * 计算目标字符串txt中每个长度为M的子串的hash值（共需要N-M-1次）
   * 比较hsh值，如果hash值不同，字符串必然不匹配，如果hash值相同，需要使用朴素算法再次判断

   hash(txt.substring(i,M)) == hash(pat)

   ```python
   Java
   //Java
   public final static int D = 256; # 字符是256进制
   public final static int Q = 9997;
   
   static int RabinKarpSerach(String txt, String pat) {
       int M = pat.length();
       int N = txt.length();
       int i, j;
       int patHash = 0, txtHash = 0;
   
       for (i = 0; i < M; i++) {
           patHash = (D * patHash + pat.charAt(i)) % Q;
           txtHash = (D * txtHash + txt.charAt(i)) % Q;
       }
   
       int highestPow = 1;  // pow(256, M-1)
       for (i = 0; i < M - 1; i++) 
           highestPow = (highestPow * D) % Q; # 找到最高位的权重值
   
       for (i = 0; i <= N - M; i++) { // 枚举起点
           if (patHash == txtHash) {
               for (j = 0; j < M; j++) {
                   if (txt.charAt(i + j) != pat.charAt(j))
                       break;
               }
               if (j == M)
                   return i;
           }
           if (i < N - M) {
               txtHash = (D * (txtHash - txt.charAt(i) * highestPow) + txt.charAt(i + M)) % Q;
               if (txtHash < 0)
                   txtHash += Q;
           }
       }
   
       return -1;
   }
   ```

   

2. KMP算法

   > [字符串匹配的KMP算法]([http://www.ruanyifeng.com/blog/2013/05/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm.html](http://www.ruanyifeng.com/blog/2013/05/Knuth–Morris–Pratt_algorithm.html))
   >
   > [KMP 字符串匹配算法视频](https://www.bilibili.com/video/av11866460?from=search&seid=17425875345653862171)

   * 找已经匹配的片段，它的最大的前缀和最大的后缀，最长有多长
   * **移动位数 = 已匹配的字符数 - 对应的部分匹配值**

   当子串与目标字符串不匹配时，其实你已经知道了前面已经匹配成功的一部分的字符（包括子串和目标字符串），设法利用已知信息，不要把‘搜索位置’移回已经比较过的位置，继续往后移，以此提高效率。 

   

3. [Boyer-Moore 算法](https://www.ruanyifeng.com/blog/2013/05/boyer-moore_string_search_algorithm.html)

   各种文本编辑器的"查找"功能（Ctrl+F），采用此算法

   * 坏字符： **后移位数 = 坏字符的位置 - 搜索词中的上一次出现位置**
   * 好后缀：**后移位数 = 好后缀的位置 - 搜索词中的上一次出现位置**
   * **Boyer-Moore算法的基本思想是，每次后移这两个规则之中的较大值。**
   * 这两个规则的移动位数，只与搜索词有关，与原字符串无关。因此，可以预先计算生成《坏字符规则表》和《好后缀规则表》。使用时，只要查表比较一下就可以了。

4. [Sunday 算法](https://blog.csdn.net/u012505432/article/details/52210975)

   **后移距离 = 最右边不重复字符到字符串末尾的距离**

   ```python
   def Sunday(str1, str2):
       if str1 == None or str2 == None or len(str1) < len(str2):
           return None
       len1, len2 = len(str1), len(str2)
       pAppear, moveDict = [], matchDict(list(str2))
       indexStr1 = 0
       while indexStr1 <= len1 - len2:
           indexStr2 = 0
           while indexStr2 < len2 and str1[indexStr1 + indexStr2] == str2[indexStr2]:
               indexStr2 += 1
           if indexStr2 == len2:
               pAppear.append(indexStr1)
               indexStr1 += len2
               continue
           if indexStr1 + len2 >= len1:
               break
           elif str1[indexStr1+len2] not in moveDict.keys():
               indexStr1 += len2 + 1
           else:
               indexStr1 += moveDict[str1[indexStr1+len2]]
       return pAppear if pAppear else False
   
   def matchDict(aList):
       moveDict = {}
       length = len(aList)
       for i in range(length-1, -1, -1):
           if aList[i] not in moveDict.keys():
               moveDict[aList[i]] = length - i
       return moveDict
   
   ```

   此外，由于BM算法需要建立一个坏字符表和一个好后缀表，Sunday算法只需要建立一个位移表(可以用hash实现，更加快速)。所以在进行字符串匹配的时候，首选Sunday算法，但是鉴于BM算法代码比较繁琐，所以KMP和BM算法就两者择其一就好。

