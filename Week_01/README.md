# 列表/数组
* 列表（其它语言称为数组），是一种基本数据类型
* 列表相关问题：
    * 列表中的元素如何存储的
    * 列表基本操作:下标查找，插入元素，删除元素。
    * 操作时间复杂度是多少
    
* pyhton列表如何实现

a[2] = 100 + 2*4

**数组与列表的不同**

       1. 数组元素类型要相同
       2. python 列表可以存不同类型元素，长度不固定，列表实际存储的是地址（地址大小是一样的（32位机器上，一个整数占4个字节，一个地址占4个字节），先拿地址，再找地址中存的元素，python内部在满了列表是，会自动复制开新的更大的列表
       3. 数组长度固定

插入删除**时间复杂度**都为O(n)

查找事件复杂度为O(1)

# 栈（FILO)
特点：后进先出LIFO
概念：栈顶，栈底

时间复杂度：插入删除均为O(1)，查找为O(n)

栈的基本操作：

       1. 进栈（压栈）：push
       2. 出栈：pop
       3. 取出栈：peek


```python
class Stack(object):
    """栈"""
    def __init__(self):
        self.items = []

    def is_empty(self):
        """判断是否为空"""
        return self.items ==[]

    def push(self,item):
        """压栈"""
        self.items.append(item)

    def pop(self):
        """出栈"""
        return self.items.pop()

    def peek(self):
        """返回栈顶元素"""
        if self.is_empty():
            return None
        return self.items[len(self.items)-1]

    def size(self):
        """返回栈顶的大小"""
        return len(self.items) 
```

## 栈的应用

### 括号匹配


```python
# 括号匹配
def brace_match(s):
    match = {'}': '{', ']': '[', ')': '('}
    stack = Stack()
    for ch in s:
        if ch in {'(', '[', '{'}:
            stack.push(ch)
        else:  # ch in {'}',')',']'}
            if stack.is_empty():
                return False
            elif stack.peek() == match[ch]:
                stack.pop()
            else:
                return False
    if stack.is_empty():
        return True
    else:
```

### 函数调用栈

### 浏览器实现前进后退




# 队列（Queue）（FIFO）
* 进行插入一端称为队尾rear，入队
* 进行删除的一端称为队头front，出队
* 先进先出 FIFO
* 增删时间复杂度均为O(1)
* 查找事件复杂度为O(n)

# Priority Queue（优先队列）

1. 插入操作O(1)
2. 取出 O(logN) 按照元素优先级顺序取出(vip先出)
3. 底层具体实现的数据结构较为多样和复杂：**heap**,**bst**,**treap**
4. 优先队列也是个接口

## python的优先队列--heapq

> [Python 的 heapq](http://docs.python.org/2/library/heapq.html)

堆是一个二叉树，它的每个父节点的值都只会小于或大于所有孩子节点（的值）。它使用了数组来实现：从零开始计数，对于所有的 *k* ，都有 `heap[k] <= heap[2*k+1]` 和 `heap[k] <= heap[2*k+2]`。 为了便于比较，不存在的元素被认为是无限大。 堆最有趣的特性在于最小的元素总是在根结点：`heap[0]`。

# heapq库的使用
```python
# heapq库的使用
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
>>>[{'names': 'cc', 'score': 100, 'height': 189}, {'names': 'dd', 'score': 80, 'height': 179}]
```



## python容器数据类型--deque

***class* `collections.deque`([*iterable*[, *maxlen*]])**

返回一个新的双向队列对象，从左到右初始化（`append()`)，从 *iterable* （迭代对象) 数据创建。如果 *iterable* 没有指定，新队列为空。

由**栈**或者**queue**生成的（”double-ended queue”）。Deque 支持线程安全，内存高效添加(append)和弹出(pop)，从两端都可以，两个方向的大概开销都是 **O(1)** 复杂度,相比于`lits`优化了定长操作和 `pop(0)` 和 `insert(0, v)` 的开销

如果 *maxlen* 没有指定或者是 `None` ，deques 可以增长到任意长度。否则，deque就限定到指定最大长度。一旦限定长度的deque满了，当新项加入时，同样数量的项就从另一端弹出。限定长度deque提供类似Unix filter `tail` 的功能。它们同样可以用与追踪最近的交换和其他数据池活动。





## 循环队列

* 队首指针前进1: front = （front+1）% MaxSize
* 队尾指针前进1: rear = (rear+1)% MaxSize
* 队空条件：rear == front
* 队满条件：（rear+1）%MaxSize == front


```python
class Queue(object):
    def __init__(self,size=100):
        self.queue = [0 for _ in range(size)]
        self.size = size
        self.rear = 0 # 队尾
        self.front = 0 # 队首，出队
        
    def push(self,element):
        if not self.is_filled(): # 判断队列是否已满
            self.rear = (self.rear+1) % self.size
            self.queue[self.rear] = element
        else:
            raise IndexError("Queue is filled")

    def pop(self):
        if not self.is_empty():
            self.front = (self.front+1)%self.size # 这一步指针往前移动了，相当于把位置腾出来了
            return self.queue[self.front]
        else:
            raise IndexError('Queue is empty.')
    

    def is_empty(self):
        """判断队空"""
        return self.rear == self.front
    
    def is_filled(self):
        """判断队满"""
        return (self.rear+1)%self.size == self.front
    
```



## 队列的内置模块
### 双端队列
* 两端都支持进队出队

### python内置


```python
from collections import deque
q = deque([1,2,3],5) # 最大长度5，队满了自动出队
q.append(1)  # 队尾进队
print(q.popleft()) # 队首出队
# 用于双向队列
# q.appendleft(1) # 队首进队
# q.pop()# 队尾出队

def tail(n):
    with open('xxxx.txt','r') as f:
        q = deque(f,n) # 相当于截取长度为n的队列，先进的文件行要是大于n就自动出列
        return q
for line in tail(5):
    print(line,end='')
```

# 链表

链表时对于数组的一种弥补，在不需要连续存储空间操作时，链表更加的灵活

* 查找时间复杂度为：O(n)

* 增删时间复杂度为：O(1)

## 链表的种类

* 单向链表
* 双向链表
* 循环链表
* 双向循环链表

### 循环链表的实现

```python
# 循环链表的核心在于链表尾部指向链表头部字节

class SingleNode(object):
    """生成节点"""
    def __init__(self,item):
        self.item = item
        self.next = None

class SinCycLinkedList(object):
    """单向循环链表"""
    def __init__(self):
        self.__head = None #定义头部节点

    def is_empty(self):
        """检查链表是否为空"""
        return self.__head is None
    def length(self):
        """返回链表的长度"""
        # 如果链表为空，则返回0
        if self.is_empty():
            return 0
        else:
            cur = self.__head
            count = 1
            while cur.next != self.__head: # 检查尾部指向不是头节点
                cur = cur.next
                count += 1
            return count

    def travel(self):
        """遍历链表"""
        if self.is_empty():
            print('This is empty')
        else:
            cur = self.__head
            print(cur.item)
            while cur.next != self.__head:
                cur = cur.next
                print(cur.item)

    def add_head(self,item):
        """头部插入"""
        new_node = SingleNode(item)
        # 判断是否为空
        if self.is_empty():
            self.__head = new_node
            new_node.next = self.__head # 自己指向自己
        else:
            new_node.next = self.__head # 首先新节点指向头节点
            cur = self.__head
            while cur.next != self.__head:
                cur = cur.next
            cur.next = new_node # 指向新节点
            # _head指向新添加的头节点
            self.__head = new_node

    def append_tail(self,item):
        """尾部插入"""
        new_node = SingleNode(item)
        # 判断是否为空
        if self.is_empty():
            self.__head = new_node
        else:
            cur = self.__head
            # 找到尾节点，然后将尾节点指向新节点
            while cur.next != self.__head:
                cur = cur.next
            cur.next = new_node
            new_node.next = self.__head # 新加入节点指向头节点

    def insert(self,pos,item):
        """指定位置插入节点"""
        # 如果位置在头节点，则头部插入
        if pos <= 0:
            self.add_head(item)
        # 如果位置在尾部，则尾部插入
        elif pos > (self.length()-1):
            self.append_tail(item)
        else:
            node = SingleNode(item)
            cur = self.__head
            count = 0
            while count < (pos-1): # 移动到指定位置前面去
                count += 1
                cur = cur.next
            node.next = cur.next
            cur.next = node

    def remove(self,item):
        """删除函数，重点需要考虑头尾部节点的删除情况"""
        # 判断是否为空
        if self.is_empty():
            return None
        cur = self.__head
        pre = None # 定义前节点

        if cur.item == item: # 如果头元素就是要找的元素
            # 如果链表不止一个节点
            if cur.next != self.__head: #也就是说头节点不是自己指向自己
                # 找到尾节点的next指向第二个节点
                while cur.next != self.__head:
                    cur = cur.next
                cur.next = self.__head.next
            else:
                 # 链表只有一个节点
                self.__head = None
        else:
            while cur.next != self.__head:
                if cur.item == item:
                    # 删除
                    pre.next = cur.next
                    return None
                else: # 向前遍历
                    pre = cur
                    cur = cur.next
            if cur.item == item: # 尾部删除情况
                pre.next = self.__head
                
    def search(self,item):
        """检查节点是否存在"""
        if self.is_empty():
            print('这是一个空链表')
            return False
        cur = self.__head
        if cur.item == item:
            return True
        while cur.next != self.__head:
            cur = cur.next
            if cur.item ==item:
                return True
        return False
```

# 跳表

# 跳表（skip list）

> 如果有序且为链表，应该如何有效加速？
>
> **升维思维+ 空间换时间**



**基于链表进行的优化**，跳表只能用于元素有序的情况，对标**平衡树**（AVL Tree）和**二分查找**，是一种 插入｜删除｜搜索都是**O(log n)**的数据结构

**优势** ：

* 原理简单
* 易实现
* 方便扩展
* 效率高
* 一些项目中用来替代平衡树，如**Redis，LevelDB等**



## 跳表如何实现

假设一个**有序**列表，查询为O(n),如何加速查询

**有序**相当于有了附加信息

​			————>**升维**（多一级信息）

## 复杂度

* 查找事件复杂度

n/2、n/4、n/8、第k级索引节点的个数为n/(2^k)，其实就相当于按顺序折半查找，类似于二分查找

假设索引有h级，最高级的索引有两个节点，n/(2^h) = 2

-------》求得h = log2(n)-1

查找事件复杂度：**log(n)**

* 空间复杂度

节点数累加（收敛）所以为**O(n)**

**O(n)**

## 现实中跳表的形态

由于元素增加和删除，导致这些索引跨的步数不相同，维护成本相对较高，增加和删除都需要更新索引，增加和删除事件复杂对会变成**O(log n)**

