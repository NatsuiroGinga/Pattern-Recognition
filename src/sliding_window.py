import pandas as pd


def sliding_window_analysis(
    df: pd.DataFrame, window_size=3, num_top_contents=50
) -> pd.DataFrame:
    """
    使用滑动窗口分析历史数据，识别当前的高频请求内容。

    本函数通过对每个内容ID的请求次数进行滑动窗口和，选取最热门的前50个内容ID。

    :param df: 包含内容ID(content_id)、时间戳(timestamp)和请求次数(request_count)的DataFrame
    :param window_size: 滑动窗口的大小，默认为3
    :param num_top_contents: 选择最热门的内容数量，默认为50

    :return: 最受欢迎的内容ID列表
    """
    # 按内容ID分组
    # 使用 pivot_table 进行聚合，避免重复记录
    pivot_df = df.pivot_table(
        index="timestamp",
        columns="content_id",
        values="request_count",
        aggfunc="sum",
        fill_value=0,
    )

    # 计算滑动窗口的总请求数
    rolling_sum = pivot_df.rolling(window=window_size).sum()

    # 获取最新时间点的滑动窗口总请求数
    latest_window = rolling_sum.iloc[-1]

    # 选择前N个高频内容
    top_contents = (
        latest_window.sort_values(ascending=False).head(num_top_contents).index.tolist()
    )

    high_freq_df = pd.DataFrame(
        {
            "content_id": top_contents,
            "total_requests": latest_window[top_contents].values,
        }
    )

    return high_freq_df
