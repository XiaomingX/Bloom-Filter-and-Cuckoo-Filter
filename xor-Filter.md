# Xor

### Xor Filter（异或过滤器）原理


Xor Filter 是由 Dmitry Vyukov 在 2019 年提出的一种概率型数据结构，专门用于高效的集合成员检测（判断元素是否属于某个集合）。它在空间效率、查询速度和实现复杂度上都优于 Bloom Filter 和 Cuckoo Filter，是三者中较新的优化方案。


#### 核心特点
- **空间效率极高**：相同假阳性率下，所需空间通常是 Bloom Filter 的 1/3 ~ 1/2，比 Cuckoo Filter 更节省内存。
- **查询速度快**：仅需 3 次哈希计算和 1 次异或运算，操作步骤固定，缓存友好。
- **支持删除**：无需复杂的置换逻辑（区别于 Cuckoo Filter），删除操作简单且无冲突。
- **局限性**：
  - 需预先知道集合大小（无法动态扩容）。
  - 插入顺序可能影响性能，需一次性批量插入（或严格按规则增量插入）。
  - 仍存在**假阳性**（但概率极低，通常低于 0.1%），无假阴性。


#### 原理详解
Xor Filter 的核心是利用**哈希函数**和**异或运算**构建指纹映射，其数据结构是一个固定大小的数组（存储“指纹”），核心逻辑如下：

1. **数据结构**：  
   一个长度为 `n` 的数组 `filter`，每个元素存储一个“指纹”（通常为 8~16 位整数），用于压缩表示元素特征。

2. **哈希函数**：  
   定义 3 个独立的哈希函数 `h0(x)`、`h1(x)`、`h2(x)`，每个函数将元素 `x` 映射到数组 `filter` 的索引（范围 `[0, n-1]`）。

3. **指纹生成**：  
   对元素 `x` 计算一个指纹 `f(x)`（通常通过哈希截取低位得到，如 16 位），用于后续验证。

4. **插入逻辑**：  
   对于元素 `x`，计算其 3 个索引 `i0=h0(x)`、`i1=h1(x)`、`i2=h2(x)`，以及指纹 `f=f(x)`。  
   插入时需满足：`filter[i0] XOR filter[i1] XOR filter[i2] = f`。  
   实现中通过调整 3 个索引中的一个位置的指纹值，使等式成立（具体需保证无冲突，通常批量插入时预处理）。

5. **查询逻辑**：  
   对元素 `x` 计算 `i0, i1, i2` 和 `f(x)`，验证 `filter[i0] XOR filter[i1] XOR filter[i2] == f(x)`。  
   - 若相等：元素“可能存在”（存在假阳性）。  
   - 若不等：元素“一定不存在”。

6. **删除逻辑**：  
   若查询确认元素“可能存在”，则重新计算 `i0, i1, i2` 和 `f(x)`，通过调整其中一个指纹值，打破 `filter[i0] XOR filter[i1] XOR filter[i2] = f(x)` 的等式，实现删除。


### Xor Filter 的 Python 实现

以下是一个简化版的 Xor Filter 实现，包含核心的插入、查询和删除功能：


```
import hashlib
import random
from typing import Generic, Hashable, List, TypeVar

T = TypeVar('T', bound=Hashable)

class XorFilter(Generic[T]):
    def __init__(self, capacity: int, fingerprint_bits: int = 16):
        """
        初始化 Xor Filter
        :param capacity: 预期存储的元素数量（需预先确定，无法动态扩容）
        :param fingerprint_bits: 指纹位数（通常 16 位，平衡空间和假阳性率）
        """
        self.capacity = capacity
        self.fingerprint_bits = fingerprint_bits
        self.max_fingerprint = (1 << fingerprint_bits) - 1  # 指纹最大值（如 16 位为 65535）
        
        # 过滤器数组大小通常为 capacity 的 1.2 倍（预留空间避免冲突）
        self.size = int(capacity * 1.2) + 3
        self.filter: List[int] = [0] * self.size  # 存储指纹的数组
        
        # 已插入元素数量
        self.count = 0

    def _hash(self, item: T, seed: int) -> int:
        """带种子的哈希函数，生成数组索引"""
        # 使用 SHA-1 哈希，结合种子生成唯一索引
        hash_obj = hashlib.sha1(f"{seed}:{item}".encode())
        return int(hash_obj.hexdigest(), 16) % self.size

    def _fingerprint(self, item: T) -> int:
        """生成元素的指纹（截取哈希低位）"""
        hash_obj = hashlib.sha1(str(item).encode())
        fp = int(hash_obj.hexdigest(), 16) % self.max_fingerprint
        return fp if fp != 0 else 1  # 避免指纹为 0（防止与初始值冲突）

    def add(self, item: T) -> bool:
        """插入元素，成功返回 True（超出容量返回 False）"""
        if self.count >= self.capacity:
            return False  # 达到容量上限
        
        f = self._fingerprint(item)
        i0 = self._hash(item, 0)
        i1 = self._hash(item, 1)
        i2 = self._hash(item, 2)
        
        # 调整三个位置中的一个指纹，使 i0^i1^i2 的异或结果等于 f
        # 简化实现：随机选择一个位置更新
        chosen_idx = random.choice([i0, i1, i2])
        current_xor = self.filter[i0] ^ self.filter[i1] ^ self.filter[i2]
        self.filter[chosen_idx] ^= (current_xor ^ f)
        
        self.count += 1
        return True

    def contains(self, item: T) -> bool:
        """查询元素是否可能存在（可能有假阳性）"""
        f = self._fingerprint(item)
        i0 = self._hash(item, 0)
        i1 = self._hash(item, 1)
        i2 = self._hash(item, 2)
        
        # 验证三个位置的异或结果是否等于指纹
        return (self.filter[i0] ^ self.filter[i1] ^ self.filter[i2]) == f

    def delete(self, item: T) -> bool:
        """删除元素（需确保元素存在，否则可能破坏过滤器）"""
        if not self.contains(item):
            return False  # 元素不存在（或假阳性）
        
        f = self._fingerprint(item)
        i0 = self._hash(item, 0)
        i1 = self._hash(item, 1)
        i2 = self._hash(item, 2)
        
        # 破坏异或等式（与插入逻辑对称）
        chosen_idx = random.choice([i0, i1, i2])
        current_xor = self.filter[i0] ^ self.filter[i1] ^ self.filter[i2]
        self.filter[chosen_idx] ^= (current_xor ^ 0)  # 用 0 打破等式
        
        self.count -= 1
        return True

    def __str__(self) -> str:
        return f"XorFilter(size={self.size}, capacity={self.capacity}, elements={self.count})"

```



### 代码说明
1. **核心参数**：  
   - `capacity`：预先设定的最大元素数（无法动态扩容）。  
   - `fingerprint_bits`：指纹位数（16 位较常用，平衡空间和假阳性率）。  

2. **哈希与指纹**：  
   - 3 个带种子的哈希函数 `_hash()` 生成元素的 3 个索引。  
   - `_fingerprint()` 生成元素的指纹（避免 0 值，防止与初始数组冲突）。  

3. **插入逻辑**：  
   通过调整 3 个索引中一个位置的指纹值，使三者异或结果等于元素指纹，确保后续查询可验证。  

4. **查询与删除**：  
   - 查询通过验证“三索引异或结果是否等于指纹”判断存在性。  
   - 删除通过打破异或等式实现（需先确认元素存在，否则可能影响其他元素）。  


### 使用示例
```python
# 初始化一个容量为 1000 的 Xor Filter
xor_filter = XorFilter(capacity=1000)

# 插入元素
xor_filter.add("apple")
xor_filter.add("banana")
xor_filter.add(123)

# 查询元素
print(xor_filter.contains("apple"))    # True
print(xor_filter.contains("orange"))   # False（大概率）
print(xor_filter.contains(123))        # True

# 删除元素
xor_filter.delete("apple")
print(xor_filter.contains("apple"))    # False（大概率）
```


### 与其他过滤器的对比
| 特性               | Bloom Filter       | Cuckoo Filter      | Xor Filter         |
|--------------------|--------------------|--------------------|--------------------|
| 空间效率           | 较低               | 中等               | 最高               |
| 查询速度           | 快（k 次哈希）     | 较快（2 次哈希）   | 最快（3 次哈希）   |
| 支持删除           | 不支持             | 支持（复杂）       | 支持（简单）       |
| 假阳性率           | 较高               | 较低               | 极低               |
| 动态扩容           | 支持（需重建）     | 有限支持           | 不支持             |
| 实现复杂度         | 简单               | 复杂（置换逻辑）   | 简单               |

Xor Filter 适合对内存敏感、需要快速查询且支持删除的场景（如缓存系统、分布式集合同步等），但需预先确定数据规模。
