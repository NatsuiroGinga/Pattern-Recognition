from sched import scheduler
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path


def get_project_root():
    """
    获取项目根目录的绝对路径。
    """
    return Path(__file__).resolve().parent.parent


class ContentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 通过全连接层
        out = self.fc(out)
        out = self.relu(out)
        return out


def train_lstm(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=50,
    batch_size=64,
    learning_rate=0.001,
    patience=10,
):
    """
    训练LSTM模型，包含早停策略，并记录训练损失和验证损失。

    :param X_train: 训练集输入数据，形状为 (样本数量, 序列长度, 特征数量)
    :param y_train: 训练集目标输出数据，形状为 (样本数量, 特征数量)
    :param X_val: 验证集输入数据
    :param y_val: 验证集目标输出数据
    :param epochs: 最大训练轮数
    :param batch_size: 批量大小
    :param learning_rate: 学习率
    :param patience: 早停耐心值，验证集损失不下降的最大轮数

    :return: 训练好的模型, train_losses_list, val_losses_list
    """
    # 创建数据集和数据加载器
    train_dataset = ContentDataset(X_train, y_train)
    val_dataset = ContentDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 获取输入和输出的维度
    input_size = X_train.shape[2]  # 特征数量
    output_size = y_train.shape[1]  # 输出特征数量

    # 定义模型
    hidden_size = 64
    num_layers = 2
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)

    # 使用 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    # 定义学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    # 早停参数初始化
    best_val_loss = float("inf")
    trigger_times = 0

    # 定义模型保存路径
    model_dir = get_project_root() / "outputs" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_lstm_model.pth"

    # 初始化损失列表
    train_losses_list = []
    val_losses_list = []

    # 训练模型
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            train_losses.append(loss.item())

        # 计算训练集的平均损失
        avg_train_loss = np.mean(train_losses)
        train_losses_list.append(avg_train_loss)

        # 验证模型
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_losses.append(loss.item())

        # 计算验证集平均损失
        avg_val_loss = np.mean(val_losses)
        val_losses_list.append(avg_val_loss)

        # 打印每个 epoch 的损失
        print(
            f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # 调度器步进
        scheduler.step(avg_val_loss)

        # 早停判断
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            # 保存最佳模型
            torch.save(model.state_dict(), model_path)
            print(f"最佳模型已保存到 {model_path}")
        else:
            trigger_times += 1
            print(f"验证损失未改善，触发次数: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("早停策略触发，停止训练。")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load(model_path))
    print(f"加载最佳模型，验证损失：{best_val_loss:.4f}")

    return model, train_losses_list, val_losses_list


def get_scaler():
    """
    加载已保存的 StandardScaler 对象。

    :return: StandardScaler 对象
    """
    scaler_path = (
        Path(__file__).resolve().parent.parent / "data" / "processed" / "scaler.pkl"
    )
    scaler = joblib.load(scaler_path)
    return scaler


def evaluate_model(model, X_test, y_test, scaler, batch_size=32):
    """
    在测试集上评估LSTM模型的性能，计算均方误差（MSE）。

    :param model: 训练好的LSTM模型
    :param X_test: 测试集输入数据，形状为 (样本数量, 序列长度, 特征数量)
    :param y_test: 测试集目标输出数据，形状为 (样本数量, 特征数量)
    :return: predictions_original (反标准化后的预测结果), mse (均方误差)
    """
    model.eval()
    predictions = []
    actuals = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loader = DataLoader(
        ContentDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    # 仅逆标准化 request_count（假设是第一个特征）
    predictions_scaled = np.zeros((predictions.shape[0], scaler.scale_.shape[0]))
    predictions_scaled[:, 0] = predictions.flatten()
    predictions_original = scaler.inverse_transform(predictions_scaled)[:, 0]

    y_test_scaled = np.zeros((actuals.shape[0], scaler.scale_.shape[0]))
    y_test_scaled[:, 0] = actuals.flatten()
    y_test_original = scaler.inverse_transform(y_test_scaled)[:, 0]

    # 四舍五入并确保非负
    predictions_integer = np.round(predictions_original).astype(int)
    predictions_integer = np.maximum(predictions_integer, 0)
    y_test_integer = np.round(y_test_original).astype(int)

    # 计算评价指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(y_test_integer, predictions_integer)
    mae = mean_absolute_error(y_test_integer, predictions_integer)

    print(f"测试集均方误差 (MSE): {mse:.4f}")
    print(f"测试集平均绝对误差 (MAE): {mae:.4f}")

    return predictions_integer, y_test_integer, mse, mae
