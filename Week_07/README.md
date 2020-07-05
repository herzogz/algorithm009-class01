## 字典树的数据结构

> 搜索引擎搜索的时候单词前缀（推断单词），词频的感应

**基本结构**

字典树，Trie树，又称单词查找树或键树，前缀树，是一种树形结构，有序树，用于保存关联数组。trie中的键通常是字符串，但也可以是其它的结构。

典型应用时用于统计和排序大量的字符串（但不仅限于字符串），所以经常被搜索引擎系统用于文本词频统计。

**优点**

最大限度减少无谓的字符串比较，**查询效率比哈希表高**。

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Trie_example.svg/2560px-Trie_example.svg.png)

* 结点本身不存完整单词
* 从根节点到某一结点，路径上经过的字符连接起来，为该节点对应的字符串
* 每个结点的所有子结点路径代表的字符都不相同
* 单词的长度为查找的深度
* 词频-----结点存储额外信息（后续可以做相应的推荐）

## 字典树的核心思想

* 空间换时间
* 利用字符串的公共前缀来降低查询时间的开销以达到提高效率的目的

## 实现Trie

其要点在于在字符串结尾的字典中添加一个终止符‘#’，以表示字符串的结束

```python
class Trie(object):
  def __init__(self):
    self.root = {}
    self.end_of_word = '#'
    
  def insert(self,word):
    node = self.root
    for char in word:
      node = node.setdefault(char,{})
    node[self.ene_of_word] = self.end_of_word
    
	def search(self,word):
    node = self.root
    for char in word:
      if char not in node:
        return False
      node = node[char]
    return self.end_of_word in node
  
 	def startsWith(self, prefix):
		node = self.root
    for char in prefix:
      if char not in node:
        return False
      node = node[char]
    return True
  
```

## 单词搜索 2

### 暴力

1. words 遍历 ---》 board search
2. O(N* m *m * 4 ^k)

### Trie

* 所有的字符-----〉放入Trie 构建起prefix
* 对于board，DFS（通过起点遍历每一个字符，DFS产生任何字符串去Trie里查询看是不是它的字串）
* 关键点：
  * Trie的构建
  * DFS定义，上下左右四个方向扩散的写法
  * 递归终止条件
* 时间复杂度$O(m*n*4*3^{k-1})$,m,n为二维数组的长款，k为单词的长度

```python
dx = [-1,1,0,0] # 创建数组，进行上下左右的扩散
dy = [0,0,-1,1]
end_of_word = '#'

class Solution(object):
  def _dfs(self,board,i,j,cur_word,cur_dict):
    cur_word += board[i][j]
    cur_dict = cur_dict[board[i][j]]
    if end_of_word in cur_dict: # terminator
      self.result.add(cur_word)
    tmp,board[i][j] = board[i][j],'@' # 使用过的字符替换成‘@’
    for k in range(4): # 上下左右扩散
      x,y = i + dx[k], j+dy[k]
      if 0 <= x <= self.m and 0 <= y <self.n\
      	and board[x][y] != '@' and board[x][y] in cur_dict:
          self._dfs(board,x,y,cur_word,cur_dict)
    board[i][j] = tmp #恢复现场，查询下一个字符串
    
  def findWords(self,board,words):
    if not board or not board[0]:
      return []
    if not words:
      return []
    self.result = set()
    
    # Trie
    root = collections.defaultdict()
    for word in words:
      node = root
      for char in word:
        node = node.setdefault(char,collections.defaultdict())
      node[end_of_word] = end_of_word
    
    self.m,self.n = len(board), lend(board[0])  # 定义长宽
    #进行遍历
    for i in range(self.m):
      for j in range(self.n):
        if board[i][j] in root: # 如果起始点在Trie中，则开始dfs
          self._dfs(board,i,j,'',root)
    return list(self.ressult)
  
  
  
```

```python
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board or not board[0]:
            return []
        if not words:
            return []
        # 构建Trie
        root = {}
        for word in words:
            node = root
            for char in word:
                node = node.setdefault(char,{})
            node['#'] = True
        
        # DFS
        def dfs(i,j,cur_node,pre,visited):
            # terminator
            if '#' in cur_node:
                res.add(pre)
            for (dx,dy) in ((-1,0),(0,1),(0,-1),(1,0)):
                x = i + dx
                y = j + dy
                if -1 < x < m and -1 < y < n and \
                    board[x][y] in cur_node and (x,y) not in visited:
                    dfs(x,y,cur_node[board[x][y]], pre + board[x][y], visited|{(x,y)})  # 使用并集，避免i，j重复
        res = set()
        m,n = len(board) , len(board[0])
        for i in range(m):
            for j in range(n):
                if board[i][j] in root:
                    dfs(i,j,root[board[i][j]], board[i][j],{(i,j)})
        return list(res)
```



#  并查集 Disjoint Set

> **并查集**是一种树型的数据结构，用于处理一些不交集（Disjoint Sets）的合并及查询问题。有一个**联合-查找算法**（**union-find algorithm**）定义了两个用于此数据结构的操作：

- `Find`：确定元素属于哪一个子集。它可以被用来确定两个元素是否属于同一子集。
- `Union`：将两个子集合并成同一个集合。

题目比较死，解决场景：

* 组团，配对问题（好友朋友圈问题，分析是否是好友）
* Group or not

## 基本操作

* **makeSet(s**): 建立一个新的并查集，其中包含s个单元素集合
* **unionSet(x,y)**：把元素x和元素y所在的集合合并，要求x和y所在的集合不相交，如果相交则不合并
* **find(x)**:找到元素x所在的集合的代表，该操作也可以用于判断两个元素是否位于同一个集合，只要将他们各自的代表比较一下就可以了

### 初始化

一开始每一个元素都有一个parent数组指向自己，表示自己就是自己的集合（代表）

### 合并｜查询

* 如何查询对于任何元素，一直往上查询它的parent，知道parent等于自己时表示找到了代表元素
* 将parent[e]----> a，或者parent[a]---> e

![截屏2020-07-05下午3.37.06](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-07-05下午3.37.06.png)

### 路径压缩

把同一条路上的所有的元素都指向代表元素，查询时间会变快（步数缩短）

![截屏2020-07-05下午3.40.10](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-07-05下午3.40.10.png)

## python实现

```python
def init(p):
  # for i = 0,...n:p[i]=i
  p = [i for i in range(n)]

def union(self,p,i,j):
  p1 = self.parent(p,i)
  p2 = self.parent(p,j)
  p[p1] = p2
  
def parent(self,p,i):
  while p[root] != root: # 找root,p[root]==root
    root = p[root]
  while p[i] != i :
    x = i
    i = p[i]
    p[x] = root
    return root
```



## 547 朋友圈问题

* 矩阵肯定为对称矩阵，主对角线均为1，`M[i][j]==M[j][i]`
* DFS,对于所有结点，把每个访问的结点扩散出去，统计为一个朋友圈
* 并查集方法
  * 创建并查集
  * 遍历矩阵，合并i,j
  * 最后看整个n里有多少不同的parent

```python
class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        if not M:
            return 0

        # 初始化并查集
        n = len(M)
        p = [i for i in range(n)]

        def _union(p,i,j):
            p1 = _parent(p,i)
            p2 = _parent(p,j)
            p[p1] = p2
        def _parent(p,i):
            root = i
            while p[root] != root: # 查询代表元素root
                root = p[root]
            while p[i] != i:
                x = i
                i = p[i]
                p[x] = root # 把所有元素指向root，压缩路径
            return root
        
        for i in range(n):
            for j in range(n):
                if M[i][j] == 1:
                    _union(p,i,j)
        return len(set(_parent(p,i) for i in range(n)))
```

小结：

* 初始化
* 再拼接（拼接过程中涉及压缩路径）
* 最后返回去重的root的个数，每个root代表一个组

# 高级搜索

## 初级搜索

1. 朴素搜索
2. 优化方法：不重复（fibonacci）、剪枝（括号生成问题）
3. 搜索方向：
   * DFS
   * BFS
   * 双向搜索：起点和终点分别做广度优先，在中间相遇
   * 启发式搜索、优先级搜索（A*）：优先队列，按照结点优先级，把更可能会达到结果的结点，先拿出来进行搜索

**搜索问题其实就是在状态树种使用各种搜索方法**，**找到最优解的问题**

## 剪枝

> 国际象棋，三子旗，五子棋

* 爬楼梯
* 括号生成

### N皇后 复习

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def DFS(queens,left_diagnal,right_diagnal):
            row = len(queens)
            if row == n: # 皇后放满了
                res.append(queens)

            for col in range(n):
                if col not in queens and row+col not in left_diagnal and row-col not in right_diagnal:
                    DFS(queens+[col],left_diagnal+[row+col],right_diagnal+[row-col])
        
        res = []
        DFS([],[],[])
        return[['.'*i+'Q'+'.'*(n-i-1) for i in cols] for cols in res]

```

### 36 有效的数独 

关键点在于如何把格子拆出来

* 行区间，列区间，3 * 3 块区间

* 使用set来判断每个区块是否含有重复的元素

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row = [[x for x in y if x != '.'] for y in board]
        col = [[x for x in y if x != '.'] for y in zip(*board)]
        pal = [[board[i+m][j+n] for m in range(3) for n in range(3) if board[i+m][j+n] != '.'] for i in (0, 3, 6) for j in (0, 3, 6)]
        return all(len(set(x)) == len(x) for x in (*row, *col, *pal))
```



### 37 解数独

方块索引 = （行 / 3）* 3 + 列 / 3

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # 初始化出每个区间可用的数字
        row = [set(range(1,10)) for _ in range(9)]
        column = [set(range(1,10)) for _ in range(9)]
        block = [set(range(1,10)) for _ in range(9)]

        empty = [] # 收集需要填数的位置
        
        # 清除已经使用的数字
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    number = int(board[i][j])
                    row[i].remove(number)
                    column[j].remove(number)
                    block[(i//3)*3 + j // 3].remove(number)
                else:
                    empty.append((i,j))  # 添加填数的位置

        def backtrack(level = 0):
            if level == len(empty): # 表示empty中所有位置处理完了
                return True
            i , j = empty[level]
            b = (i // 3) * 3 + j // 3
            for val in row[i] & column[j] & block[b]: # 遍历剩余可用的数字
                row[i].remove(val)
                column[j].remove(val)
                block[b].remove(val)
                board[i][j] = str(val)
                if backtrack(level + 1):
                    return True
                # 如果行不通，恢复现场
                row[i].add(val)
                column[j].add(val)
                block[b].add(val)
            return False
        backtrack()

```



## 双向BFS

![截屏2020-07-05下午8.41.42](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-07-05下午8.41.42.png)

### 127 单词接龙

```python
import string
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if not beginWord or not endWord or not wordList:
            return 0
        if endWord not in wordList:
            return 0
        
        step = 1
        begin = {beginWord}
        end = {endWord}

        n = len(beginWord)
        wordlist = set(wordList)

        while begin:
            step += 1
            new_begin = set()
            for word in begin:
                for i in range(n):
                    for char in string.ascii_lowercase:
                        if char != word[i]:
                            new_word = word[:i] + char + word[i+1:]
                            if new_word in end: # 与反向的扩散相遇
                                return step # 转换已完成，返回步数
                            if new_word in wordlist:
                                new_begin.add(new_word)
                                wordlist.remove(new_word)
            begin = new_begin
            if len(end) < len(begin): # 交换方向，更小的优先搜索
                begin , end = end, begin
        return 0
```

### 433 最小基因变化

```python
class Solution:
    def minMutation(self, start: str, end: str, bank: List[str]) -> int:
        bank = set(bank)

        if not end or not start or end not in bank:
            return -1
        
        change = {
            'A':"CGT",
            'C':"AGT",
            'G':'ACT',
            'T':'ACG'
        }
        queue = [(start,0)]

        while queue:
            node,step = queue.pop(0)
            if node == end:
                return step # 说明变化完成，返回步数

            for i , v in enumerate(node):
                for c in change[v]: # 和单词转化异曲同工，只是这里时基因的转换
                    new = node[:i] + c + node[i+1:]

                    if new in bank:
                        queue.append((new,step+1))
                        bank.remove(new)
        return -1

```



## 启发式搜索（A*）

> Heuristic Search，启发式搜索基于BFS

将BFS中的队列转化成优先队列

```python
def AstarSearch(graph,start,end):
  pq = collections.priority_queue() # 优先级-》估价函数
  pq.append([start])
  visited.add(start)
  
  while pq:
    node = pq.pop() # 智能化，定义优先级（根据问题）
    visited.add(node)
    
    process(node)
    nodes = generate_related_nodes(node)
    unvisited = [node for node in nodes if node not in visited]
    pq.push(unvisited)
  
```

### 估价函数

启发式函数：h(n)，它用来评价哪些结点最有希望是我们要找的结点，h(n)会返回一个非负实数，也可以认为是从结点n的目标结点路径的估计成本。

启发式函数是一种**告知搜索方向**的方法，它提供一种明智的方法来猜测哪个邻居结点会导向一个目标。

### 1091 二进制矩阵中的最短路径

**BFS**

```python
class Solution:
  def shortestPathBinaryMatrix(self,grid):
    q, n = [(0,0,2)],len(grid)
    if grid[0][0] or grid[-1][-1]:
      return -1
    elif n<=2:
      return n
    for i , j , d in q:
      # 当前结点 i， j，距离 d
      for x, y in [(i - 1, j - 1),(i - 1, j),(i - 1, j + 1),(i, j - 1),(i, j + 1),(i + 1, j - 1),(i + 1, j),(i + 1, j + 1)]: # 八联通
        if 0 <= x< n and 0 <= y < n and not grid[x][y]:
          if x == y == n - 1:
            return d
          q += [(x, y , d + 1)] # 把下一个候选者放到q里
          grid[x][y] = 1 # 走过的路赋值为1， 不再重复计算
     return -1
```

**A*** 

曼哈顿距离（坐标与坐标之间的坐标差绝对值的和），距离越近，扩散越好

[国际站参考]('https://leetcode.com/problems/shortest-path-in-binary-matrix/discuss/313347/A*-search-in-Python')

### 773 滑动谜题

**BFS**

```python
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        # 定义0板块挪动的方向
        moves = {
            0:[1,3],
            1:[0,2,4],
            2:[1,5],
            3:[0,4],
            4:[1,3,5],
            5:[2,4]
        }

        visited = set()
        step = 0
        s = ''.join(str(c) for row in board for c in row)
        q = [s] # 初始状态
        
        while q:
            new = []
            for ss in q:
                visited.add(s)
                if ss == '123450':
                    return step
                arr = [c for c in ss]
                # 开始移动0
                zero_index = ss.index('0')
                for move in moves[zero_index]:
                    new_arr = arr[:]
                    new_arr[zero_index] , new_arr[move] = new_arr[move] , new_arr[zero_index]
                    new_s = ''.join(new_arr)
                    if new_s not in visited:
                        new.append(new_s)
            step += 1
            q = new
        return  -1


```

```python
class Solution:
    def slidingPuzzle(self, board: List[List[int]]) -> int:
    board = board[0] + board[1]
    moves = [(1,3),(0,2,4),(1,5),(0,4),(1,3,5),(2,4)]
    
    q, visited = [(tuple(board),board.index(0),0)],set()
    
    while q:
      state, now, step = q.pop(0)
      if state == (1,2,3,4,5,0):
        return step # 换好了
      for next in moves[now]:
        new_state = list(state)
        new_state[next],new_state[now] = new_state[now],new_state[next]
        new_state = tuple[new_state]
        if new_state not in visited:
          q.append((new_state,next,step+1))
       visited.add(state)
    return -1
      
```



# AVL树和红黑树的实现和特性

对于二叉搜索树，在极端情况下退化为链表，查找时间复杂度变高，于是为了保证性能，我们需要保证二维的维度：

* 左右子树结点平衡（平衡二叉树）
  * AVL
  * 红黑树
  * treap
  * splay 伸展树
  * B+ 数据库索引
  * 2-3 tree

## AVL树

* Balance Factor（平衡因子）：
  * 左子树高度减去右子树高度（有时相反）： banlance factor = {-1,0,1}
  * 所有叶子结点高度为0
  * 始终保证所有结点平衡因子为{-1,0,1}中
  * 当平衡因子超出这个范围时，我们将进行旋转的操作
* 旋转操作来进行平衡（4种）
  * 左旋： 右右子树---》 左旋
  * 右旋： 做作子树---〉 右旋
  * 左右旋： 左右子树---》 左右旋
  * 右左旋： 右左子树---〉 右左旋
* 带有子树的旋转

[动画参考![截屏2020-07-05下午11.04.26](/Users/echotrees/Library/Application Support/typora-user-images/截屏2020-07-05下午11.04.26.png)]('https://zhuanlan.zhihu.com/p/63272157')

* **不足之处：结点需要存储额外信息，且调整次数频繁**

## 红黑树

> 近似平衡二叉树（调整次数少一些），它能够确保每个结点的左右子树的**高度差小于两倍**

* 每个结点要么红色，要么黑色
* 根节点时黑色
* 每个叶结点（NIL结点，空结点）是黑色的
* 不能有相邻接的两个红色结点
* 从任一结点到每个叶子结点的所有路径都包含相同数目的黑色结点

![upload.wikimedia.org/wikipedia/commons/thumb/6/...](https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Red-black_tree_example.svg/450px-Red-black_tree_example.svg.png)

### 关键性质

**从根到叶子的最长的可能路径不多余最短的可能路径的两倍长**



## 对比

* AVL树比红黑树查找更快，因为前者更加严格平衡
* 红黑树提供更快的插入删除操作，因为AVL旋转操作会更多
* AVL要存的额外信息更多，需要额外内存更多，红黑树对额外内存空间消耗更小
* 读操作用AVL（Database）， 插入和删除用红黑树



