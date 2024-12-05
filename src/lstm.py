import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTMModel 类是一个简单的 LSTM 模型。
    """

    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.input_size = config.num_contents
        self.hidden_size = config.hidden_size
        self.output_size = config.num_contents
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.num_linear_layers = config.num_linear_layers

        # 定义两层 LSTM
        self.lstm1 = nn.LSTM(
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.lstm2 = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        # 定义全连接层列表
        self.fcs = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.num_linear_layers - 1)
            ]
            + [nn.Linear(self.hidden_size, self.output_size)]
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # LSTM 前向传播
        out, (hn, cn) = self.lstm1(x, (h0, c0))
        out = self.relu(out)
        out, _ = self.lstm2(out, (hn, cn))
        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 全连接层前向传播
        for fc in self.fcs[:-1]:
            out = fc(out)
            out = self.relu(out)

        return self.fcs[-1](out)
