from calendar import c
from typing import Tuple
import numpy as np

from src.common.function import sigmoid


class LSTM:
    """
    LSTM类

    Attributes:
        params: 参数列表, [Wx, Wh, b]
        grads: 梯度列表
        cache: 缓存, 保存正向传播的中间结果，它们将在反向传播的计算中使用。
    """

    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(
        self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        正向传播

        Params:
            x: 输入数据
            h_prev: 前一时刻的隐藏状态.
            c_prev: 前一时刻的细胞状态

        Returns:
            (h_next, c_next): 下一时刻的隐藏状态和细胞状态
        """
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        # Affine
        # x: (N, D), Wx: (D, 4H), h_prev: (N, H), Wh: (H, 4H), b: (4H,)
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        # slice
        # forget gate
        f = A[:, :H]
        # input value
        g = A[:, H : 2 * H]
        # input gate
        i = A[:, 2 * H : 3 * H]
        # output gate
        o = A[:, 3 * H :]

        # Activation function
        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)

        return h_next, c_next
