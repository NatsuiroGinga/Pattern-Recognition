import numpy as np
import pandas as pd
import os
from pathlib import Path

from semver import process


def get_project_root():
    """
    获取项目根目录的 Path 对象。
    """
    return Path(__file__).resolve().parent.parent


def generate_synthetic_data(
    num_contents=500, num_experiments=1000, time_span=24, seed=42
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
    for hour in range(time_span):
        requests = np.random.multinomial(n=num_experiments, pvals=zipf_dist)
        for content_id, req_count in zip(content_ids, requests):
            data.append(
                {
                    "timestamp": f"2024-12-01 {hour}:00",
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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据预处理
    """
    # 1. 处理时间戳，转换为datetime类型
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # 2. 按时间排序
    df = df.sort_values(by="timestamp")
    # TODO: 可以添加更多预处理步骤，如平滑处理、去除异常值等
    # 3. 保存处理后的数据
    processed_dir = get_project_root() / "data" / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(os.path.join(processed_dir, "processed_requests.csv"), index=False)
    return df
