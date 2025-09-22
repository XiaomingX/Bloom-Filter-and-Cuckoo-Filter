### Bloom Filter 与 Cuckoo Filter 原理


#### Bloom Filter（布隆过滤器）
Bloom Filter 是一种空间高效的概率型数据结构，用于快速判断一个元素是否属于某个集合。其核心特点是：
- **优势**：空间效率极高，插入和查询时间复杂度为 O(k)（k 为哈希函数数量）。
- **局限性**：存在**假阳性**（错误判断元素存在），但无**假阴性**（不会错误判断元素不存在）；不支持删除操作。

**原理**：
1. 初始化一个长度为 m 的位数组（bit array），所有位初始化为 0。
2. 定义 k 个独立的哈希函数，每个函数将元素映射到位数组的一个索引。
3. **插入**：对元素应用 k 个哈希函数，将对应的 k 个比特位设为 1。
4. **查询**：对元素应用 k 个哈希函数，若所有对应比特位均为 1，则元素“可能存在”；否则“一定不存在”。


#### Cuckoo Filter（布谷鸟过滤器）
Cuckoo Filter 是 Bloom Filter 的改进版，支持删除操作，且通常具有更低的假阳性率。其核心特点是：
- **优势**：支持插入、查询、删除操作；假阳性率更低。
- **局限性**：插入可能失败（当过滤器满时）；实现较复杂。

**原理**：
1. 使用一个或多个哈希表（通常为 2 个），每个桶（bucket）可存储多个元素（通常为 4-8 个）。
2. 每个元素通过两个哈希函数生成两个可能的存储位置（h1(x) 和 h2(x)）。
3. **插入**：
   - 若两个位置中有一个桶有空闲空间，直接放入。
   - 若均满，则随机踢出其中一个位置的元素，将新元素放入，并将被踢出的元素重新哈希到其另一个位置（h2(x) 或 h1(x)），重复此过程。
   - 若置换次数超过阈值，判定过滤器满，插入失败。
4. **查询**：检查两个哈希位置的桶中是否包含该元素。
5. **删除**：从两个哈希位置的桶中移除该元素（需存储元素指纹以避免误删）。


### Python 实现代码
```
# bloom_filter.py
import math
import random
from typing import Generic, Hashable, TypeVar

T = TypeVar('T', bound=Hashable)

class BloomFilter(Generic[T]):
    def __init__(self, capacity: int, false_positive_rate: float = 0.01):
        """
        初始化布隆过滤器
        :param capacity: 预期存储的元素数量
        :param false_positive_rate: 可接受的假阳性率
        """
        # 计算位数组长度 m
        self.m = int(-(capacity * math.log(false_positive_rate)) / (math.log(2) ** 2)) + 1
        # 计算哈希函数数量 k
        self.k = int((self.m / capacity) * math.log(2)) + 1
        # 初始化位数组（用整数模拟，节省空间）
        self.bit_array = 0  # 初始为0，所有位均为0

    def _hash(self, item: T, seed: int) -> int:
        """带种子的哈希函数，生成不同的哈希值"""
        # 使用内置hash结合种子，确保不同种子生成不同哈希
        return (hash(item) ^ seed) % self.m

    def add(self, item: T) -> None:
        """插入元素"""
        for seed in range(self.k):
            pos = self._hash(item, seed)
            # 将第pos位设为1
            self.bit_array |= 1 << pos

    def contains(self, item: T) -> bool:
        """查询元素是否可能存在"""
        for seed in range(self.k):
            pos = self._hash(item, seed)
            # 检查第pos位是否为1，若有任何一位为0则不存在
            if (self.bit_array & (1 << pos)) == 0:
                return False
        return True

    def __str__(self) -> str:
        return f"BloomFilter(m={self.m}, k={self.k}, bits_used={bin(self.bit_array).count('1')})"

```
```
# cuckoo_filter.py
import hashlib
from typing import Generic, Hashable, List, TypeVar

T = TypeVar('T', bound=Hashable)

class CuckooFilter(Generic[T]):
    def __init__(self, capacity: int, bucket_size: int = 4, max_swaps: int = 500):
        """
        初始化布谷鸟过滤器
        :param capacity: 预期存储的元素数量
        :param bucket_size: 每个桶可存储的元素数量
        :param max_swaps: 最大置换次数（超过则判定为满）
        """
        # 桶数量（通常为 capacity / bucket_size 的1.2倍，预留空间）
        self.num_buckets = int(capacity / bucket_size * 1.2) + 1
        self.bucket_size = bucket_size  # 每个桶的容量
        self.max_swaps = max_swaps      # 最大置换次数
        # 初始化桶（二维列表：num_buckets 个桶，每个桶最多 bucket_size 个元素）
        self.buckets: List[List[int]] = [[] for _ in range(self.num_buckets)]

    def _fingerprint(self, item: T) -> int:
        """生成元素的指纹（用于存储和比对，减少冲突）"""
        # 使用MD5哈希生成16位指纹（可调整长度）
        hash_bytes = hashlib.md5(str(item).encode()).digest()
        return int.from_bytes(hash_bytes[:2], byteorder='big')  # 16位指纹

    def _hash1(self, item: T) -> int:
        """第一个哈希函数：映射到桶索引"""
        return hash(item) % self.num_buckets

    def _hash2(self, fingerprint: int, idx1: int) -> int:
        """第二个哈希函数：基于指纹和第一个索引计算第二个位置"""
        return (idx1 ^ hash(fingerprint)) % self.num_buckets

    def add(self, item: T) -> bool:
        """插入元素，成功返回True，失败（过滤器满）返回False"""
        fingerprint = self._fingerprint(item)
        idx1 = self._hash1(item)
        idx2 = self._hash2(fingerprint, idx1)

        # 尝试直接放入两个位置中的一个
        if len(self.buckets[idx1]) < self.bucket_size:
            self.buckets[idx1].append(fingerprint)
            return True
        if len(self.buckets[idx2]) < self.bucket_size:
            self.buckets[idx2].append(fingerprint)
            return True

        # 两个位置都满，随机选择一个位置开始置换
        current_idx = random.choice([idx1, idx2])
        current_fp = fingerprint

        # 尝试置换
        for _ in range(self.max_swaps):
            # 随机踢出桶中的一个元素
            kick_idx = random.randint(0, self.bucket_size - 1)
            current_fp, self.buckets[current_idx][kick_idx] = self.buckets[current_idx][kick_idx], current_fp
            # 计算被踢出元素的另一个位置
            current_idx = self._hash2(current_fp, current_idx)
            # 尝试放入新位置
            if len(self.buckets[current_idx]) < self.bucket_size:
                self.buckets[current_idx].append(current_fp)
                return True

        # 超过最大置换次数，插入失败
        return False

    def contains(self, item: T) -> bool:
        """查询元素是否可能存在"""
        fingerprint = self._fingerprint(item)
        idx1 = self._hash1(item)
        idx2 = self._hash2(fingerprint, idx1)
        # 检查两个位置的桶中是否有该指纹
        return fingerprint in self.buckets[idx1] or fingerprint in self.buckets[idx2]

    def delete(self, item: T) -> bool:
        """删除元素，成功返回True，不存在返回False"""
        fingerprint = self._fingerprint(item)
        idx1 = self._hash1(item)
        idx2 = self._hash2(fingerprint, idx1)

        # 从桶中移除指纹
        if fingerprint in self.buckets[idx1]:
            self.buckets[idx1].remove(fingerprint)
            return True
        if fingerprint in self.buckets[idx2]:
            self.buckets[idx2].remove(fingerprint)
            return True
        return False

    def __str__(self) -> str:
        total = sum(len(bucket) for bucket in self.buckets)
        return f"CuckooFilter(buckets={self.num_buckets}, bucket_size={self.bucket_size}, elements={total})"

```

下面分别实现 Bloom Filter 和 Cuckoo Filter 的核心功能：

### 代码说明

1. **Bloom Filter**：
   - 核心参数：`capacity`（预期元素数）和 `false_positive_rate`（可接受假阳性率），自动计算位数组长度 `m` 和哈希函数数量 `k`。
   - 哈希函数：通过种子区分不同哈希，确保多样性。
   - 操作：`add()` 插入元素，`contains()` 检查元素是否可能存在。


2. **Cuckoo Filter**：
   - 核心参数：`capacity`（预期元素数）、`bucket_size`（桶容量）、`max_swaps`（最大置换次数）。
   - 指纹机制：使用16位指纹减少存储开销和冲突。
   - 双哈希函数：`_hash1()` 计算初始位置，`_hash2()` 基于指纹和初始位置计算第二个位置。
   - 操作：`add()` 插入（支持置换）、`contains()` 查询、`delete()` 删除。


### 使用示例

```python
# 测试 Bloom Filter
bloom = BloomFilter(capacity=1000, false_positive_rate=0.01)
bloom.add("apple")
bloom.add("banana")
print(bloom.contains("apple"))   # True
print(bloom.contains("orange"))  # False（大概率）

# 测试 Cuckoo Filter
cuckoo = CuckooFilter(capacity=1000)
cuckoo.add("dog")
cuckoo.add("cat")
print(cuckoo.contains("dog"))    # True
print(cuckoo.delete("dog"))      # True
print(cuckoo.contains("dog"))    # False
```

两种过滤器均适用于需要高效空间和快速查询的场景，Bloom Filter 更简单，Cuckoo Filter 支持删除且假阳性率更低。
