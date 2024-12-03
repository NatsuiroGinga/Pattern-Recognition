from typing import List
import pandas as pd


def sliding_window_prediction(
    df: pd.DataFrame, window_size=3, num_top_contents=50
) -> List[int]:
    """
    使用滑动窗口方法预测内容的未来请求次数，并返回最受欢迎的内容列表。

    本函数通过对每个内容ID的请求次数进行滑动窗口平均，预测下一时段的请求次数。
    
    然后根据预测值对内容进行排序，选取最热门的前50个内容ID。

    :param df: 包含内容ID(content_id)、时间戳(timestamp)和请求次数(request_count)的DataFrame
    :param window_size: 滑动窗口的大小，默认为3
    :param num_top_contents: 选择最热门的内容数量，默认为50

    :return: 最受欢迎的内容ID列表
    """
    # 按内容ID分组
    grouped = df.groupby("content_id")
    predictions = {}
    for content_id, group in grouped:
        group = group.sort_values(by="timestamp")
        # 使用滑动窗口计算移动平均
        group["rolling_avg"] = group["request_count"].rolling(window=window_size).mean()
        # 预测下一时段的请求次数, 即滑动窗口的最后一个值
        predictions[content_id] = group["rolling_avg"].iloc[-1]
    # 根据预测值排序，选择热门内容
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    # 取前50个热门内容
    top_contents = [item[0] for item in sorted_predictions[:num_top_contents]]
    return top_contents
