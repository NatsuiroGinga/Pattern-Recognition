from typing import Tuple
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path

from panel import cache
from sklearn.discriminant_analysis import StandardScaler

from src.sliding_window import sliding_window_analysis


def get_project_root():
    """
    获取项目根目录的 Path 对象。
    """
    return Path(__file__).resolve().parent.parent


def generate_synthetic_data(
    num_contents=500,
    num_experiments=1000,
    time_span=24,
    seed=42,
    start_time="2024-12-01 00:00",
) -> pd.DataFrame:
    """
    生成合成数据

    模拟用户内容请求行为。使用Zipf分布来模拟内容访问热度。

    :param num_contents: 内容数量
    :param num_experiments: 实验次数，即每个时间段的请求次数
    :param time_span: 时间跨度, 默认为24小时
    :param seed: 随机种子
    """
    np.random.seed(seed)
    # 1. 生成内容ID
    content_ids = np.arange(1, num_contents + 1)
    # 2. 生成访问热度，遵循Zipf分布
    ranks = np.arange(1, num_contents + 1)
    zipf_dist = 1 / ranks
    zipf_dist /= zipf_dist.sum()
    # 3. 生成每个内容的每个时间段的请求次数
    data = []

    # 4. 生成连续的时间戳
    try:
        # 4.1 尝试将 start_time 转换为 pandas 的 Timestamp 对象
        start_timestamp = pd.to_datetime(start_time)
    except Exception as e:
        raise ValueError(f"start_time 参数格式错误: {e}")

    timestamps = pd.date_range(start=start_timestamp, periods=time_span, freq="h")

    for timestamp in timestamps:
        requests = np.random.multinomial(n=num_experiments, pvals=zipf_dist)
        for content_id, req_count in zip(content_ids, requests):
            data.append(
                {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
                    "content_id": content_id,
                    "request_count": req_count,
                }
            )
    df = pd.DataFrame(data)
    # 4. 保存原始数据
    raw_dir = get_project_root() / "data" / "raw"
    os.makedirs(raw_dir, exist_ok=True)
    df.to_csv(os.path.join(raw_dir, "synthetic_requests.csv"), index=False)
    print(f"合成数据已保存到 {raw_dir}")

    return df


def preprocess_data(
    df: pd.DataFrame, window_size=3, cache_size=1000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    数据预处理

    :param df: 包含内容ID(content_id)、时间戳(timestamp)和请求次数(request_count)的DataFrame
    :param window_size: 滑动窗口大小, 默认为3, 求过去3个时间段的请求次数和
    :param cache_size: 缓存大小, 默认为1000

    :return: 预处理后的数据和高频内容ID列表
    """
    # 1. 处理时间戳，转换为datetime类型
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # 2. 按时间排序
    df = df.sort_values(by="timestamp")
    # TODO: 可以添加更多预处理步骤，如平滑处理、去除异常值等
    # 3. 使用滑动窗口识别高频内容
    high_freq_contents = sliding_window_analysis(
        df, window_size, num_top_contents=cache_size
    )
    # 4. 保存处理后的数据
    processed_dir = get_project_root() / "data" / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(os.path.join(processed_dir, "processed_requests.csv"), index=False)
    print(f"预处理后的数据已保存到 {processed_dir}")
    # 5. 保存高频内容
    high_freq_dir = processed_dir / "high_freq_contents"
    high_freq_dir.mkdir(parents=True, exist_ok=True)
    high_freq_file_path = high_freq_dir / "high_freq_contents.csv"
    high_freq_contents.to_csv(high_freq_file_path, index=False)
    print(f"高频内容已保存到 {high_freq_file_path}")

    return df, high_freq_contents


def prepare_lstm_data(df: pd.DataFrame, sequence_length=10):
    """
    准备LSTM模型所需的数据格式。

    :param df: 预处理后的 DataFrame
    :param sequence_length: 序列长度

    :return: X, y 数据
    """
    # 选择高频内容
    processed_dir = get_project_root() / "data" / "processed"
    high_freq_dir = get_project_root() / "data" / "processed" / "high_freq_contents"
    high_freq_file_path = high_freq_dir / "high_freq_contents.csv"
    high_freq_contents = pd.read_csv(high_freq_file_path)
    top_contents = high_freq_contents["content_id"].tolist()

    # 过滤高频内容
    df_high = df[df["content_id"].isin(top_contents)]

    # 创建 pivot 表
    pivot_df = df_high.pivot(
        index="timestamp", columns="content_id", values="request_count"
    ).fillna(0)

    print("lstm数据")
    print(pivot_df.head())

    # 归一化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_df)

    # 保存 scaler
    scaler_path = processed_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler 已保存到 {scaler_path}")

    # 构建序列数据
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i : i + sequence_length])
        y.append(scaled_data[i + sequence_length])

    X = np.array(X)
    y = np.array(y)

    print(f"LSTM 输入数据形状：X={X.shape}, y={y.shape}")

    return X, y, scaler
