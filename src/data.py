import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

from config import Config

np.random.seed(42)


class DataGenerator:
    """
    DataGenerator 类是一个数据生成器，用于生成模拟数据集并将其转换为 PyTorch DataLoader 对象。

    Args:
        config (Config): 配置对象
    """

    def __init__(self, config: Config):
        self.num_contents = config.num_contents
        self.time_steps = config.time_steps
        self.popular_size = config.popular_size
        self.sequence_length = config.sequence_length
        self.raw_data_path = config.raw_data_path
        self.df = None
        self.X = None
        self.y = None
        self.scaler = MinMaxScaler()

    def generate_data(self):
        popular_contents = np.random.choice(
            self.num_contents, size=self.popular_size, replace=False
        )
        data = []
        for _ in range(self.time_steps):
            content_requests = np.random.poisson(lam=5, size=self.num_contents)
            content_requests[popular_contents] += np.random.poisson(lam=20)
            data.append(content_requests)
        data = np.array(data)
        self.df = pd.DataFrame(
            data, columns=[f"Content_{i}" for i in range(self.num_contents)]
        )
        # 保存到data/raw_data.csv
        self.df.to_csv(self.raw_data_path, index=False)

    def preprocess_data(self):
        X = []
        y = []
        for i in range(len(self.df) - self.sequence_length):
            X.append(self.df.iloc[i : i + self.sequence_length].values)
            y.append(self.df.iloc[i + self.sequence_length].values)
        X = np.array(X)
        y = np.array(y)
        # 数据归一化
        X_shape = X.shape
        X = X.reshape(-1, self.num_contents)
        X = self.scaler.fit_transform(X)
        X = X.reshape(X_shape)
        y = self.scaler.transform(y)
        # 转换为张量
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        # 保存
        torch.save(self.X, "data/X.pt")
        torch.save(self.y, "data/y.pt")

    def get_data_loaders(self, batch_size, train_ratio, val_ratio, test_ratio):
        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "训练集、验证集和测试集的比例之和必须为1"

        dataset = TensorDataset(self.X, self.y)
        total_size = len(dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)

        # 获取所有索引
        indices = list(range(total_size))
        np.random.shuffle(indices)

        # 划分索引
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        # 保存索引
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        # 创建数据集
        train_dataset = TensorDataset(self.X[train_indices], self.y[train_indices])
        val_dataset = TensorDataset(self.X[val_indices], self.y[val_indices])
        test_dataset = TensorDataset(self.X[test_indices], self.y[test_indices])

        # 创建数据加载器
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader
