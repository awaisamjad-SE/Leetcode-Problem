# 🔥 COMPLETE LEETCODE PROBLEM TYPES & SOLUTION GUIDE
## Master ALL 30+ Problem Categories | Solve ANY LeetCode Problem | Never Forget Solutions

---

# 📚 TABLE OF CONTENTS

## PART 1: PROBLEM TYPES & CATEGORIES
1. [Overview: All Problem Types](#overview-all-problem-types)
2. [Category 1: Arrays & Strings](#category-1-arrays--strings)
3. [Category 2: Hash Maps & Sets](#category-2-hash-maps--sets)
4. [Category 3: Two Pointers](#category-3-two-pointers)
5. [Category 4: Sliding Window](#category-4-sliding-window)
6. [Category 5: Stacks & Queues](#category-5-stacks--queues)
7. [Category 6: Linked Lists](#category-6-linked-lists)
8. [Category 7: Trees & BSTs](#category-7-trees--bsts)
9. [Category 8: Graphs](#category-8-graphs)
10. [Category 9: Heaps & Priority Queues](#category-9-heaps--priority-queues)
11. [Category 10: Binary Search](#category-10-binary-search)
12. [Category 11: Dynamic Programming](#category-11-dynamic-programming)
13. [Category 12: Backtracking](#category-12-backtracking)
14. [Category 13: Bit Manipulation](#category-13-bit-manipulation)
15. [Category 14: Math & Geometry](#category-14-math--geometry)
16. [Category 15: Tries](#category-15-tries)
17. [Category 16: Union Find](#category-16-union-find)
18. [Category 17: Greedy](#category-17-greedy)
19. [Category 18: Intervals](#category-18-intervals)
20. [Category 19: Design/OOP](#category-19-designoop)
21. [Additional Advanced Categories](#additional-advanced-categories)

## PART 2: HOW TO SOLVE ANY PROBLEM
22. [Golden Method to Solve ANY Problem](#golden-method-to-solve-any-problem)
23. [Decision Tree](#decision-tree)
24. [Keyword Recognition System](#keyword-recognition-system)
25. [Memory Techniques](#memory-techniques)
26. [Practice Strategy](#practice-strategy)

---

# OVERVIEW: ALL PROBLEM TYPES

## How Many Types?

**Answer:** ~25-35 main types, but they can overlap.

**Why Memorizing Doesn't Work:**
- 10,000+ LeetCode problems exist
- You can't memorize them all
- But they all fall into categories

**What Works:**
- Learn 25-35 problem types
- Learn solution patterns for each
- Practice recognizing which type a problem is
- Apply the right technique

---

## Distribution of Problems:

```
Easy   (1400+ problems)  → 35-40% Arrays, Hash Map, String
Medium (4000+ problems)  → Trees, DP, Graphs, BFS/DFS
Hard   (2000+ problems)  → DP, Graph algorithms, Complex logic

Popularity:
Most Asked: Hash Map, DP, Trees, Graphs, Two Pointers
Less Asked:  Bit Manipulation, Union Find, Trie
```

---

## The 25+ Problem Categories:

```
Data Structure Based:
  1. Arrays & Strings
  2. Hash Maps & Sets
  3. Linked Lists
  4. Stacks & Queues
  5. Trees & BSTs
  6. Graphs
  7. Heaps & Priority Queues
  8. Tries
  9. Union Find

Algorithm Based:
  10. Two Pointers
  11. Sliding Window
  12. Binary Search
  13. DFS & Recursion
  14. BFS & Level Order
  15. Dynamic Programming
  16. Backtracking
  17. Greedy
  18. Topological Sort

Special Topics:
  19. Bit Manipulation
  20. Math & Number Theory
  21. Geometry
  22. String Matching/Regex
  23. Design & OOP
  24. Intervals
  25. Divide & Conquer
  
Advanced:
  26. Segment Tree
  27. Monotonic Stack
  28. Prefix Sum
  29. Matrix DP
  30. Game Theory
```

---

# CATEGORY 1: ARRAYS & STRINGS

## What is it?
Problems involving manipulation of arrays and strings.

## Recognizing This Category:

**Keywords in Problem:**
```
- "Array" or "List"
- "Subarray" or "Substring"
- "Index" or "Position"
- "Rotate", "Reverse", "Swap"
- "Rearrange", "Shuffle"
- "Find max/min in array"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Best Time to Buy & Sell Stock | Easy |
| Container With Most Water | Medium |
| Remove Duplicates | Easy |
| Rotate Array | Medium |
| Product of Array Except Self | Medium |
| Best Time Buy Sell II | Medium |

## Key Techniques:

```
1. In-place modification
2. Two-pointer approach
3. Prefix/Suffix arrays
4. Space-time trade-offs
```

## Template Solutions:

### Template 1: Find Max/Min

```python
def findMax(arr):
    max_val = float('-inf')
    for num in arr:
        max_val = max(max_val, num)
    return max_val
```

### Template 2: Two Pointer

```python
def twoPointerArray(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        # Process
        left += 1
        right -= 1
```

### Template 3: Prefix/Suffix

```python
def prefixSumArray(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix
```

## Examples:

**Problem: Best Time to Buy & Sell Stock**
```python
def maxProfit(prices):
    min_price = prices[0]
    max_profit = 0
    
    for price in prices[1:]:
        profit = price - min_price
        max_profit = max(max_profit, profit)
        min_price = min(min_price, price)
    
    return max_profit
```

**Problem: Rotate Array**
```python
def rotate(nums, k):
    k = k % len(nums)
    nums.reverse()
    nums[:k] = reversed(nums[:k])
    nums[k:] = reversed(nums[k:])
```

---

# CATEGORY 2: HASH MAPS & SETS

## What is it?
Problems involving frequency counting, lookup, or grouping.

## Recognizing This Category:

**Keywords:**
```
- "Count", "Frequency"
- "Duplicate", "Unique"
- "Pair", "Two numbers add up"
- "Anagram"
- "Intersection", "Union"
- "First unique"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Two Sum | Easy |
| Contains Duplicate | Easy |
| Valid Anagram | Easy |
| Group Anagrams | Medium |
| Top K Frequent | Medium |
| Majority Element | Easy |

## Key Techniques:

```
1. Hash Map for frequency
2. Hash Set for uniqueness
3. Counter for easy counting
4. defaultdict for grouping
```

## Template Solutions:

### Template 1: Frequency Counting

```python
def countFrequency(arr):
    freq = {}
    for num in arr:
        freq[num] = freq.get(num, 0) + 1
    return freq

# Or use Counter
from collections import Counter
freq = Counter(arr)
```

### Template 2: Two Sum Pattern

```python
def twoSum(arr, target):
    seen = {}
    for i, num in enumerate(arr):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
    return []
```

### Template 3: Grouping

```python
def groupByKey(arr):
    groups = {}
    for item in arr:
        key = item['category']
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    return groups
```

## Examples:

**Problem: Group Anagrams**
```python
def groupAnagrams(strs):
    groups = {}
    for word in strs:
        # Sort word to get key
        key = ''.join(sorted(word))
        if key not in groups:
            groups[key] = []
        groups[key].append(word)
    return list(groups.values())
```

**Problem: Top K Frequent Elements**
```python
def topKFrequent(nums, k):
    freq = Counter(nums)
    return [num for num, _ in freq.most_common(k)]
```

---

# CATEGORY 3: TWO POINTERS

## What is it?
Using two pointers to traverse data efficiently.

## Recognizing This Category:

**Keywords:**
```
- "Sorted array"
- "Pair of numbers"
- "Reverse", "Palindrome"
- "Remove duplicates"
- "Merge"
- "One pointer is slower/faster"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Two Sum II (Sorted) | Easy |
| Valid Palindrome | Easy |
| Reverse String | Easy |
| Container With Most Water | Medium |
| 3Sum | Medium |

## Key Techniques:

```
1. Opposite direction (left ↔ right)
2. Same direction (slow → fast)
3. Fast & slow pointers
```

## Template Solutions:

### Template 1: Opposite Direction

```python
def twoPointerOpposite(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        if arr[left] + arr[right] == target:
            return [arr[left], arr[right]]
        elif arr[left] + arr[right] < target:
            left += 1
        else:
            right -= 1
```

### Template 2: Same Direction

```python
def twoPointerSame(arr):
    slow = 0
    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]
    return slow + 1
```

---

# CATEGORY 4: SLIDING WINDOW

## What is it?
Maintaining a window that slides over data.

## Recognizing This Category:

**Keywords:**
```
- "Subarray", "Substring"
- "Contiguous"
- "Maximum/Minimum in range"
- "Longest/Shortest substring"
- "At most K", "Exactly K"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Longest Substring Without Repeat | Medium |
| Max Consecutive Ones | Easy |
| Sliding Window Maximum | Hard |
| Minimum Window Substring | Hard |

## Template Solution:

```python
def slidingWindow(s, target):
    left = 0
    window_sum = 0
    result = 0
    
    for right in range(len(s)):
        window_sum += s[right]
        
        while window_sum > target and left <= right:
            window_sum -= s[left]
            left += 1
        
        result = max(result, right - left + 1)
    
    return result
```

---

# CATEGORY 5: STACKS & QUEUES

## What is it?
LIFO (Stack) or FIFO (Queue) data structures.

## Recognizing This Category:

**Keywords for Stack:**
```
- "Valid parentheses", "Matching brackets"
- "Undo/Redo"
- "Next greater/smaller"
- "Monotonic stack"
- "DFS simulation"
```

**Keywords for Queue:**
```
- "BFS"
- "Level order"
- "Circular queue"
- "Task scheduling"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Valid Parentheses | Easy |
| Min Stack | Medium |
| Daily Temperatures | Medium |
| Trapping Rain Water II | Hard |

## Template Solutions:

### Stack: Valid Parentheses

```python
def isValid(s):
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
        else:
            stack.append(char)
    
    return len(stack) == 0
```

### Queue: BFS

```python
from collections import deque

def bfs(start, graph):
    queue = deque([start])
    visited = {start}
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited
```

---

# CATEGORY 6: LINKED LISTS

## What is it?
Nodes connected by pointers.

## Recognizing This Category:

**Keywords:**
```
- "Linked List"
- "Node pointer"
- "Reverse list"
- "Detect cycle"
- "Merge lists"
- "Middle element"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Reverse Linked List | Easy |
| Detect Cycle | Easy |
| Merge Two Sorted Lists | Easy |
| Remove Nth Node From End | Medium |
| LRU Cache | Medium |

## Key Techniques:

```
1. Fast & slow pointers
2. Dummy node
3. Three pointers for reversal
4. Hash set for cycle detection
```

## Template Solutions:

### Template 1: Reverse

```python
def reverseList(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

### Template 2: Detect Cycle

```python
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

### Template 3: Find Middle

```python
def findMiddle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

---

# CATEGORY 7: TREES & BSTs

## What is it?
Hierarchical data structures.

## Recognizing This Category:

**Keywords:**
```
- "Tree", "Binary Tree", "BST"
- "Traversal" (inorder, preorder, postorder)
- "Path sum"
- "Lowest Common Ancestor"
- "Validate BST"
- "Height", "Balance"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Maximum Depth | Easy |
| Invert Tree | Easy |
| Path Sum | Easy |
| Lowest Common Ancestor | Medium |
| Serialize/Deserialize | Hard |

## Template Solutions:

### Template 1: DFS Traversal

```python
def dfs(node):
    if not node:
        return
    
    # Preorder: process, left, right
    process(node)
    dfs(node.left)
    dfs(node.right)
```

### Template 2: BFS Level Order

```python
from collections import deque

def levelOrder(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    
    return result
```

### Template 3: Path Sum

```python
def hasPathSum(root, target, current_sum=0):
    if not root:
        return False
    
    current_sum += root.val
    
    if not root.left and not root.right:
        return current_sum == target
    
    return (hasPathSum(root.left, target, current_sum) or
            hasPathSum(root.right, target, current_sum))
```

---

# CATEGORY 8: GRAPHS

## What is it?
Nodes (vertices) and edges representing relationships.

## Recognizing This Category:

**Keywords:**
```
- "Graph", "Node", "Edge"
- "Connected components"
- "Topological sort"
- "Shortest path"
- "DFS", "BFS"
- "Cycle detection"
- "Bipartite"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Number of Islands | Medium |
| Course Schedule | Medium |
| Alien Dictionary | Hard |
| Network Delay Time | Medium |

## Template Solutions:

### Template 1: DFS

```python
def dfs(node, graph, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(neighbor, graph, visited)
```

### Template 2: BFS

```python
from collections import deque

def bfs(start, graph):
    queue = deque([start])
    visited = {start}
    
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return visited
```

### Template 3: Topological Sort (Kahn's Algorithm)

```python
def topologicalSort(graph, n):
    indegree = [0] * n
    adj = [[] for _ in range(n)]
    
    # Build graph
    for u in graph:
        for v in graph[u]:
            adj[u].append(v)
            indegree[v] += 1
    
    # Find nodes with indegree 0
    queue = deque([i for i in range(n) if indegree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in adj[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == n else []
```

---

# CATEGORY 9: HEAPS & PRIORITY QUEUES

## What is it?
Data structure that efficiently retrieves min or max.

## Recognizing This Category:

**Keywords:**
```
- "Top K", "Kth largest"
- "Min heap", "Max heap"
- "Priority queue"
- "Median"
- "Merge K lists"
- "Closest points"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Kth Largest Element | Easy |
| Top K Frequent | Medium |
| Merge K Sorted Lists | Hard |
| Find Median Data Stream | Hard |

## Template Solutions:

```python
import heapq

# Min Heap
heap = []
heapq.heappush(heap, 5)
min_val = heapq.heappop(heap)

# Max Heap (negate values)
max_heap = []
heapq.heappush(max_heap, -5)

# Heap from list
arr = [3, 1, 4, 1, 5]
heapq.heapify(arr)

# Top K
def topK(arr, k):
    return heapq.nlargest(k, arr)
```

---

# CATEGORY 10: BINARY SEARCH

## What is it?
Searching in sorted data by dividing in half.

## Recognizing This Category:

**Keywords:**
```
- "Sorted array"
- "Find first/last position"
- "Search in rotated array"
- "Binary search on answer"
- "Peak element"
- "Find in matrix"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Binary Search | Easy |
| First Bad Version | Easy |
| Find First & Last Position | Medium |
| Search in Rotated Array | Medium |

## Template Solutions:

### Template 1: Basic Binary Search

```python
def binarySearch(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

### Template 2: Find First Position

```python
def findFirst(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

### Template 3: Binary Search on Answer

```python
def binarySearchOnAnswer(n, max_val):
    # Find minimum value x such that condition(x) is True
    left, right = 0, max_val
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        if condition(mid):
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    
    return result
```

---

# CATEGORY 11: DYNAMIC PROGRAMMING

## What is it?
Breaking problem into overlapping subproblems and storing results.

## Recognizing This Category:

**Keywords:**
```
- "Optimal", "Maximum", "Minimum"
- "Count ways"
- "Can you achieve"
- "Best", "Worst"
- "Unique paths"
- "House robber", "Coin change"
- "Edit distance"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Climbing Stairs | Easy |
| House Robber | Medium |
| Coin Change | Medium |
| Longest Increasing Subsequence | Medium |
| Edit Distance | Hard |
| Palindrome Partitioning | Hard |

## Key Insight:

**DP = Recursion + Memoization**

```
1. Define state: dp[i] = answer for subproblem i
2. Base case: dp[0], dp[1]
3. Recurrence: dp[i] = function of previous states
4. Order: Solve in dependency order
```

## Template Solutions:

### Template 1: Climbing Stairs (1D DP)

```python
def climbStairs(n):
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
```

### Template 2: House Robber (1D DP)

```python
def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    dp = [0] * len(nums)
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    
    return dp[-1]
```

### Template 3: 0/1 Knapsack (2D DP)

```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],
                    dp[i-1][w-weights[i-1]] + values[i-1]
                )
            else:
                dp[i][w] = dp[i-1][w]
    
    return dp[n][capacity]
```

---

# CATEGORY 12: BACKTRACKING

## What is it?
Trying all possible solutions by exploring and undoing.

## Recognizing This Category:

**Keywords:**
```
- "All combinations"
- "All permutations"
- "All subsets"
- "Find all solutions"
- "Sudoku", "N-Queens"
- "Restore IP addresses"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Combinations | Medium |
| Permutations | Medium |
| Subsets | Medium |
| N-Queens | Hard |
| Sudoku Solver | Hard |

## Template Solution:

```python
def backtrack(path, remaining, result):
    # Base case: found a solution
    if not remaining:
        result.append(path[:])  # Add copy of path
        return
    
    # Try each choice
    for choice in remaining:
        # Make choice
        path.append(choice)
        new_remaining = remaining - {choice}
        
        # Recurse
        backtrack(path, new_remaining, result)
        
        # Undo choice (backtrack)
        path.pop()
```

---

# CATEGORY 13: BIT MANIPULATION

## What is it?
Solving problems using bitwise operations.

## Recognizing This Category:

**Keywords:**
```
- "XOR", "AND", "OR"
- "Single number"
- "Power of 2"
- "Bit operations"
- "Count 1s"
- "Missing number"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Single Number | Easy |
| Power of Two | Easy |
| Missing Number | Easy |
| Number of 1 Bits | Easy |

## Key Tricks:

```
a ^ a = 0
a ^ 0 = a
n & (n-1) = remove rightmost 1 bit
n | (n+1) = set rightmost 0 to 1
```

## Template Solutions:

```python
# XOR all: n ^ n = 0
def singleNumber(nums):
    result = 0
    for num in nums:
        result ^= num
    return result

# Count 1 bits
def countBits(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

# Check power of 2
def isPowerOfTwo(n):
    return n > 0 and (n & (n-1)) == 0
```

---

# CATEGORY 14: MATH & GEOMETRY

## What is it?
Mathematical and geometric problem solving.

## Recognizing This Category:

**Keywords:**
```
- "Prime number"
- "GCD", "LCM"
- "Factorial"
- "Distance", "Angle"
- "Coordinates"
- "Fraction"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Reverse Integer | Medium |
| String to Integer | Medium |
| Palindrome Number | Easy |
| Plus One | Easy |

## Template Solutions:

```python
# Prime number check
def isPrime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# GCD (Euclidean algorithm)
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Count digits
def countDigits(n):
    count = 0
    while n > 0:
        count += 1
        n //= 10
    return count
```

---

# CATEGORY 15: TRIES

## What is it?
Tree structure for prefix-based string searching.

## Recognizing This Category:

**Keywords:**
```
- "Autocomplete"
- "Word search"
- "Prefix matching"
- "Dictionary"
- "Starts with"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Implement Trie | Medium |
| Word Search II | Hard |
| Autocomplete | Medium |

## Template Solution:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

---

# CATEGORY 16: UNION FIND

## What is it?
Tracking connected components and relationships.

## Recognizing This Category:

**Keywords:**
```
- "Connected components"
- "Is connected"
- "Union"
- "Friends of friends"
- "Cycle detection"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Number of Connected Components | Easy |
| Accounts Merge | Hard |
| Redundant Connection | Medium |

## Template Solution:

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        
        if px == py:
            return False
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

---

# CATEGORY 17: GREEDY

## What is it?
Making locally optimal choices hoping for global optimum.

## Recognizing This Category:

**Keywords:**
```
- "Best choice at each step"
- "Activity selection"
- "Minimize", "Maximize"
- "Interval scheduling"
- "Huffman coding"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Jump Game | Medium |
| Interval Scheduling | Medium |
| Meeting Rooms | Medium |

## Template Solution:

```python
def greedy_approach(intervals):
    # Sort by end time
    intervals.sort(key=lambda x: x[1])
    
    selected = [intervals[0]]
    last_end = intervals[0][1]
    
    for start, end in intervals[1:]:
        if start >= last_end:
            selected.append((start, end))
            last_end = end
    
    return selected
```

---

# CATEGORY 18: INTERVALS

## What is it?
Problems involving ranges or intervals.

## Recognizing This Category:

**Keywords:**
```
- "Interval", "Range"
- "Merge intervals"
- "Overlap"
- "Meeting rooms"
- "Insert interval"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| Merge Intervals | Medium |
| Insert Interval | Medium |
| Interval List Intersections | Medium |

## Template Solution:

```python
def mergeIntervals(intervals):
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort()
    result = [intervals[0]]
    
    for current in intervals[1:]:
        last = result[-1]
        
        if current[0] <= last[1]:
            # Overlapping, merge
            result[-1] = (last[0], max(last[1], current[1]))
        else:
            # Non-overlapping, add new
            result.append(current)
    
    return result
```

---

# CATEGORY 19: DESIGN/OOP

## What is it?
Designing data structures or systems.

## Recognizing This Category:

**Keywords:**
```
- "Design", "Implement"
- "LRU Cache"
- "Min Stack"
- "Serialization"
- "Object-oriented"
```

## Common Problems:

| Problem | Difficulty |
|---------|------------|
| LRU Cache | Hard |
| Min Stack | Easy |
| Serialize/Deserialize | Hard |

## Template: LRU Cache

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return -1
        
        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        
        self.cache[key] = value
        
        if len(self.cache) > self.capacity:
            # Remove least recently used
            self.cache.popitem(last=False)
```

---

# ADDITIONAL ADVANCED CATEGORIES

## Category 20: SEGMENT TREE

```
Used for: Range sum queries, range updates
Difficulty: Hard
Time: O(log n)
```

## Category 21: MONOTONIC STACK

```
Used for: Next greater element, histogram problems
Difficulty: Medium
Key Pattern: Maintain stack in increasing/decreasing order
```

## Category 22: PREFIX SUM

```
Used for: Range sum, subarray sum
Difficulty: Easy/Medium
Template: prefix[i] = prefix[i-1] + arr[i]
```

## Category 23: MATRIX DP

```
Used for: Unique paths, min path sum, longest path
Difficulty: Medium/Hard
2D DP table where dp[i][j] = answer for position (i,j)
```

## Category 24: GAME THEORY

```
Used for: Predict winner, game outcomes
Difficulty: Hard
Key: Think recursively about optimal play
```

---

# GOLDEN METHOD TO SOLVE ANY PROBLEM

## 5-Step Framework (Never Forget!)

### STEP 1️⃣: UNDERSTAND THE PROBLEM (2 minutes)

```
❓ Questions to Ask:
- What is the input?
- What is the output?
- What are the constraints?
- What are edge cases?
- Can I modify the input?
- What's the time/space limit?
```

**Never skip this step!** Most mistakes come from misunderstanding.

---

### STEP 2️⃣: IDENTIFY THE CATEGORY (1 minute)

Use the **Keyword Recognition System** below to identify which category:

```
IF "sum of two numbers" AND "array" → Hash Map (Two Sum)
IF "subarray" OR "substring" → Sliding Window
IF "sorted" AND "search" → Binary Search
IF "tree" AND "traversal" → Trees/DFS/BFS
IF "all combinations" OR "all permutations" → Backtracking
IF "count", "frequency" → Hash Map
IF "first unique", "duplicate" → Hash Set
IF "merge lists" → Linked List
IF "optimal", "maximum", "minimum" → DP
IF "graph" AND "connected" → Graph/BFS/DFS
```

---

### STEP 3️⃣: CHOOSE DATA STRUCTURE & APPROACH (2 minutes)

Based on category, pick the right approach:

| Category | Approach | Data Structure |
|----------|----------|-----------------|
| Arrays | Two Pointers / Sliding Window | Array |
| Hash Map | Hash Map lookup | dict |
| Two Pointers | Opposite/Same direction | None |
| Sliding Window | Expand/Contract | dict |
| Stacks | LIFO processing | list |
| Queues | FIFO processing | deque |
| Linked List | Pointer manipulation | ListNode |
| Trees | DFS/BFS | TreeNode |
| Graphs | DFS/BFS | Graph dict |
| Heaps | Priority handling | heapq |
| Binary Search | Divide and conquer | None |
| DP | State tracking | 1D/2D array |
| Backtracking | Try all | Recursion |

---

### STEP 4️⃣: DRY RUN WITH EXAMPLE (3 minutes)

**ALWAYS simulate with small input before coding!**

```
Example: Two Sum
Input: nums = [2, 7], target = 9

Dry Run:
- i=0: num=2, need=7, seen={}
  - Add 2 to seen: seen={2:0}
- i=1: num=7, need=2, is 2 in seen? YES!
  - Return [0, 1] ✅
```

**If dry run fails, fix logic before coding!**

---

### STEP 5️⃣: CODE & TEST (5-10 minutes)

Use template from the category:

```python
# Template structure:
def solve(input):
    # Step 1: Initialize
    result = ...
    
    # Step 2: Main logic
    for ... in ...:
        # Process
        pass
    
    # Step 3: Return
    return result

# Test with provided examples
```

---

## MEMORY TRICK: The 5 Steps

```
🧠 UICDCT
  U = Understand
  I = Identify category
  C = Choose approach
  D = Dry run
  C = Code
  T = Test
```

Remember: **U-I-C-D-C-T**

---

# DECISION TREE

```
START: Read Problem

│
├─ Is it about FINDING/COUNTING something?
│  ├─ Frequency/Count? → Hash Map
│  ├─ Two numbers/pairs? → Hash Map or Two Pointers
│  ├─ All combinations/subsets? → Backtracking
│  └─ Specific element? → Binary Search or Hash Map
│
├─ Is it about MANIPULATING something?
│  ├─ Array/String? → Two Pointers, Sliding Window
│  ├─ List? → Linked List techniques
│  ├─ Tree? → DFS/BFS
│  └─ Graph? → DFS/BFS, Union Find
│
├─ Is it about OPTIMIZING something?
│  ├─ Max/Min value? → DP or Greedy
│  ├─ Shortest path? → BFS or Dijkstra
│  └─ Top K? → Heap
│
├─ Is it about CHECKING something?
│  ├─ Validity? → Stack or Recursion
│  ├─ Connection? → Union Find or BFS
│  └─ Existence? → Hash Set or Hash Map
│
└─ Is it SPECIAL?
   ├─ Design pattern? → OOP/Design
   ├─ Bit operations? → Bit Manipulation
   ├─ String matching? → Trie or Regex
   └─ Math? → Math formula

END: Choose approach
```

---

# KEYWORD RECOGNITION SYSTEM

## REMEMBER: Don't try to memorize problems, memorize KEYWORDS!

### Complete Keyword Map:

```
ARRAYS:
  "array", "index", "sort", "rotate", "shuffle"
  → Two Pointers, Sorting, Prefix Sum

HASH MAP:
  "count", "frequency", "duplicate", "unique"
  "pair", "two numbers", "group", "anagram"
  → Use dict, Counter, defaultdict

STRINGS:
  "substring", "subarray", "contiguous"
  "palindrome", "reverse", "longest/shortest"
  → Sliding Window, Two Pointers, DP

STACKS:
  "valid", "parentheses", "matching"
  "next greater/smaller", "monotonic"
  → Use stack (list)

QUEUES:
  "level order", "BFS", "scheduling"
  → Use deque

LINKED LISTS:
  "node", "pointer", "reverse", "cycle"
  "merge", "middle", "remove nth"
  → Two pointers, dummy node

TREES:
  "tree", "binary", "BST", "traversal"
  "path", "sum", "ancestor", "height"
  → DFS, BFS, Recursion

GRAPHS:
  "graph", "node", "edge", "connected"
  "component", "path", "cycle", "topological"
  → DFS, BFS, Union Find

HEAPS:
  "top K", "kth largest", "min/max"
  "median", "merge"
  → heapq

BINARY SEARCH:
  "sorted", "search", "find first/last"
  "peak", "rotated"
  → Binary search template

DYNAMIC PROGRAMMING:
  "optimal", "maximum", "minimum"
  "count ways", "can achieve", "best"
  "house robber", "coin change"
  → DP with state definition

BACKTRACKING:
  "all combinations", "all permutations"
  "all subsets", "all solutions"
  "Sudoku", "N-Queens"
  → Recursion with backtrack

BIT MANIPULATION:
  "XOR", "AND", "OR"
  "single number", "power of 2"
  "bit", "count 1s"
  → Bitwise operations

INTERVALS:
  "interval", "range", "overlap"
  "merge", "meeting rooms"
  → Sort and merge

GREEDY:
  "best choice", "maximize", "minimize"
  "activity selection"
  → Greedy selection
```

---

# MEMORY TECHNIQUES

## Technique 1: ACRONYM METHOD

Remember patterns by acronyms:

```
Two Sum:     HM (Hash Map)
Palindrome:  EP (Expand around center or math)
Two Pointer: OP (Opposite pointers)
Sliding Win: SM (Slide and maintain)
Stack:       LP (LIFO/Parentheses)
Tree:        DB (DFS/BFS)
Graph:       DB (DFS/BFS)
DP:          SR (State + Recurrence)
Backtrack:   RT (Recurse and backtrack)
```

## Technique 2: VISUAL ANCHOR

Create mental images:

```
Two Sum: Imagine two people searching from opposite ends
Sliding Window: Imagine a moving window over array
Stack: Imagine stacking plates
Queue: Imagine waiting in line
Linked List: Imagine chain of nodes
Tree: Imagine upside-down tree with root at top
Graph: Imagine city with roads connecting places
DP: Imagine climbing stairs (each step uses previous)
Backtracking: Imagine exploring maze, going back when stuck
```

## Technique 3: PROBLEM FAMILY GROUPING

Group similar problems:

```
SAME PATTERN - TWO SUM FAMILY:
  - Two Sum (array, target)
  - 3Sum (array, target)
  - 4Sum (array, target)
  - Two Sum II (sorted array)
  Solution: Hash Map or Two Pointers

SAME PATTERN - SUBARRAY FAMILY:
  - Max Subarray Sum
  - Min Subarray Sum
  - Subarray Sum equals K
  - Max Subarray with K distinct
  Solution: Sliding Window or Prefix Sum or DP

SAME PATTERN - TREE FAMILY:
  - Tree Traversal
  - Path Sum
  - Lowest Common Ancestor
  - Serialize/Deserialize
  Solution: DFS/BFS or Recursion

SAME PATTERN - DP FAMILY:
  - Climbing Stairs
  - House Robber
  - Coin Change
  - Unique Paths
  Solution: DP with state definition
```

## Technique 4: "WHY THIS WORKS" UNDERSTANDING

For each pattern, understand WHY it works:

```
Hash Map for Two Sum:
  WHY: We need O(1) lookup of complement
  REASON: Array traversal is O(n), hash lookup is O(1)
  RESULT: Total O(n) instead of O(n²)

Sliding Window:
  WHY: Avoid recalculating same subarray
  REASON: Move window boundaries instead of recalculating
  RESULT: O(n) instead of O(n²)

Binary Search:
  WHY: Sorted data has order property
  REASON: Can eliminate half search space each time
  RESULT: O(log n) instead of O(n)

DP:
  WHY: Overlapping subproblems
  REASON: Store results to reuse
  RESULT: Polynomial time instead of exponential

DFS/BFS for Graphs:
  WHY: Systematic traversal of all nodes
  REASON: Avoid revisiting nodes with visited set
  RESULT: Can find paths, cycles, components
```

---

# PRACTICE STRATEGY

## The Science of Not Forgetting:

### Phase 1: RECOGNITION (Week 1)

**Goal:** Learn to identify problem categories

```
Day 1-2: Read about 3 categories
  - Arrays & Strings
  - Hash Maps & Sets
  - Two Pointers

Day 3-4: Read about 3 more categories
  - Sliding Window
  - Stacks & Queues
  - Linked Lists

Day 5-6: Read about 3 more categories
  - Trees
  - Graphs
  - Binary Search

Day 7: Review - Can you identify 9 categories?
```

**Test yourself:** I give you a problem, you identify the category in 10 seconds.

---

### Phase 2: SOLUTION TEMPLATES (Week 2)

**Goal:** Learn solutions for each category

```
For each of 25 categories:
  1. Read the template
  2. Understand WHY it works
  3. Type the template 3 times
  4. Modify template for different input

Example: Hash Map Template
  Write standard version
  Write for Counter
  Write for defaultdict
```

**Memorization trick:** Create a "cheat sheet" with all 25 templates.

---

### Phase 3: PROBLEM SOLVING (Week 3-4)

**Goal:** Solve actual problems

```
Strategy: Solve by category

Week 3:
  Arrays:       Solve 5 easy problems
  Hash Map:     Solve 5 easy problems
  Two Pointers: Solve 5 easy problems
  Sliding Win:  Solve 5 easy problems

Week 4:
  Stack/Queue:  Solve 5 easy + 3 medium
  Linked List:  Solve 5 easy + 3 medium
  Trees:        Solve 5 easy + 3 medium
  Graphs:       Solve 3 medium
  DP:           Solve 3 medium
```

**For each problem:**
  1. Identify category (10 sec)
  2. Recall template (10 sec)
  3. Dry run (2 min)
  4. Code (5 min)
  5. Test (1 min)

---

### Phase 4: MIXING (Week 5+)

**Goal:** Solve random problems (like real interviews)

```
Do 100 random problems from all categories:
  - 40 Easy
  - 40 Medium
  - 20 Hard

Track which ones you get wrong → Review that category
```

---

## The 80/20 Rule:

```
Most Common Interview Problems (by category):

80% of interviews use:
  1. Hash Map (20%)      → Two Sum, Anagram, etc.
  2. DP (20%)            → Climbing Stairs, Coin Change
  3. Trees (15%)         → Traversal, Path Sum
  4. Graphs (15%)        → Islands, Connections
  5. Two Pointers (10%)  → Reverse, Merge

20% of interviews use:
  - Stacks, Queues, Bit Manipulation, etc.

Focus on the 80% first!
```

---

## Daily Practice Routine (30 min):

```
Day 1-7 (Week 1): Recognition
  5 min: Learn 1 category
  10 min: Identify category in 5 random problems
  10 min: Review memory techniques
  5 min: Create mental image

Week 2: Templates
  15 min: Learn 1 template deep
  10 min: Type template 3 times from memory
  5 min: Modify template

Week 3-4: Solving
  5 min: Read problem
  10 min: Dry run
  10 min: Code
  5 min: Test

Week 5+: Mixing
  5 min: Random problem
  10 min: Identify category + recall approach
  10 min: Solve
  5 min: Verify
```

---

# SUMMARY: NEVER FORGET FRAMEWORK

## The Master Framework:

```
📊 CATEGORIES (25+)
  ↓
  ├─ Arrays & Strings
  ├─ Hash Maps & Sets
  ├─ Two Pointers
  ├─ Sliding Window
  ├─ Stacks & Queues
  ├─ Linked Lists
  ├─ Trees & BSTs
  ├─ Graphs
  ├─ Heaps
  ├─ Binary Search
  ├─ Dynamic Programming
  ├─ Backtracking
  ├─ Bit Manipulation
  ├─ Math
  ├─ Tries
  ├─ Union Find
  ├─ Greedy
  ├─ Intervals
  ├─ Design
  └─ ... (more)

🎯 FOR EACH CATEGORY:
  ├─ Keywords to recognize it
  ├─ Common problems
  ├─ Solution template
  └─ Time/Space complexity

🔧 TO SOLVE ANY PROBLEM:
  1. Understand (questions?)
  2. Identify (which category?)
  3. Choose (which approach?)
  4. Dry run (does it work?)
  5. Code (implement template)

🧠 REMEMBER TECHNIQUES:
  ├─ Acronyms (HM, EP, OP)
  ├─ Visual anchors (mental images)
  ├─ Family grouping (2Sum family)
  └─ "Why it works" understanding
```

---

## The Promise:

If you follow this framework:

✅ You will **recognize** any problem type in 10 seconds
✅ You will **remember** the solution approach
✅ You will **solve** the problem in minutes (not hours)
✅ You will **pass** coding interviews

---

## Next Steps:

1. **Week 1:** Read all 25 categories (one per day if possible)
2. **Week 2:** Learn templates for each category
3. **Week 3-4:** Solve 50+ problems by category
4. **Week 5+:** Solve random mixed problems

After 4-5 weeks of consistent practice:
- You'll recognize any problem instantly
- You'll remember solutions without looking them up
- You'll be ready for any interview

---

## The Real Secret:

**It's not about memorizing 10,000 problems.**

**It's about understanding 25 categories and their templates.**

Once you master the categories, every problem becomes a simple application of the pattern you already know.

---

**You've got this! 🚀 Start with one category today!**

