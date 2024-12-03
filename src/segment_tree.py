class SegmentTree:
    def __init__(self, data, func):
        """
        初始化线段树。
        :param data: 输入数据列表。
        :param func: 区间统计函数，如sum, min, max等。
        """
        self.n = len(data)
        self.func = func
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [0] * (2 * self.size)
        self.build(data)

    def build(self, data):
        # 将数据放入叶子节点
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        # 填充剩余的叶子节点
        for i in range(self.size + self.n, 2 * self.size):
            self.tree[i] = 0  # 根据func不同，可能需要调整默认值
        # 从下往上构建内部节点
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.func(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, index, value):
        """
        更新索引处的值，并更新相关节点。
        :param index: 要更新的索引。
        :param value: 新值。
        """
        index += self.size
        self.tree[index] = value
        while index > 1:
            index >>= 1
            self.tree[index] = self.func(self.tree[2 * index], self.tree[2 * index + 1])

    def query(self, left, right):
        """
        查询区间 [left, right) 的统计值。
        :param left: 左边界（包含）。
        :param right: 右边界（不包含）。
        :return: 区间统计值。
        """
        res = None
        left += self.size
        right += self.size
        while left < right:
            if left & 1:
                res = (
                    self.tree[left] if res is None else self.func(res, self.tree[left])
                )
                left += 1
            if right & 1:
                right -= 1
                res = (
                    self.tree[right]
                    if res is None
                    else self.func(res, self.tree[right])
                )
            left >>= 1
            right >>= 1
        return res
