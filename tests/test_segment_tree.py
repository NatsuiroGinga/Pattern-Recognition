import sys
import os
import unittest

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.segment_tree import SegmentTree


class TestSegmentTree(unittest.TestCase):
    def test_sum_query(self):
        data = [1, 3, 5, 7, 9, 11]
        st = SegmentTree(data, func=lambda a, b: a + b)
        self.assertEqual(st.query(0, 3), 1 + 3 + 5)  # 9
        self.assertEqual(st.query(2, 5), 5 + 7 + 9)  # 21
        self.assertEqual(st.query(0, 6), sum(data))  # 36

    def test_update(self):
        data = [1, 3, 5, 7, 9, 11]
        st = SegmentTree(data, func=lambda a, b: a + b)
        st.update(3, 10)  # 修改第4个元素 from 7 to 10
        self.assertEqual(st.query(0, 6), 1 + 3 + 5 + 10 + 9 + 11)  # 39
        self.assertEqual(st.query(3, 4), 10)  # 单个元素


if __name__ == "__main__":
    unittest.main()
