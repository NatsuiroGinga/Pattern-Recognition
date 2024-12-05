import os
import numpy as np
from sklearn.metrics import mean_squared_error
import torch.nn as nn
from torch.utils.data import DataLoader
import torch

from config import Config


class ModelTrainer:
    """
    ModelTrainer 类是一个模型训练器，用于训练 PyTorch 模型。

    Args:
        model (nn.Module): PyTorch 模型
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        config (Config): 配置对象
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )
        self.num_epochs = config.num_epochs
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = np.inf  # 初始化最佳验证损失
        self.patience = config.patience  # 早停的容忍度
        self.min_delta = config.min_delta  # 最小改进量
        self.early_stop = False
        self.counter = 0  # 记录没有改进的 epoch 数
        self.best_model_state = None  # 保存最佳模型的状态
        self.save_path = config.best_model_path  # 模型保存路径

    def train(self):
        for epoch in range(self.num_epochs):
            if self.early_stop:
                print("早停触发，停止训练")
                break
            # ======== 训练阶段 ========
            self.model.train()
            train_loss = 0
            for inputs, targets in self.train_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            # ======== 验证阶段 ========
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(self.val_loader)
            self.val_losses.append(avg_val_loss)

            print(
                f"Epoch [{epoch+1}/{self.num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}"
            )
            # ======== 早停和保存最佳模型 ========
            if avg_val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = avg_val_loss
                self.counter = 0
                # 保存最佳模型
                self.best_model_state = self.model.state_dict()
                torch.save(self.best_model_state, self.save_path)
                print(
                    f"验证损失改善，保存当前模型到 {self.save_path}，最佳验证损失: {self.best_val_loss:.6f}"
                )
            else:
                self.counter += 1
                print(f"验证损失未改善 ({self.counter}/{self.patience})")
                if self.counter >= self.patience:
                    self.early_stop = True
                    print("达到早停阈值，停止训练")

    def evaluate(self, X, y, scaler):
        # 加载最佳模型
        if os.path.exists(self.save_path):
            self.model.load_state_dict(torch.load(self.save_path))
            print(f"从 {self.save_path} 加载最佳模型进行评估")
        elif self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("加载最佳模型进行评估")
        else:
            print("未保存最佳模型，使用最后的模型进行评估")

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X)
        y_true = y.numpy()
        y_pred = y_pred.numpy()
        # 反归一化
        y_true = scaler.inverse_transform(y_true)
        y_pred = scaler.inverse_transform(y_pred)
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        print(f"MSE: {mse:.2f}")
        return y_true, y_pred
