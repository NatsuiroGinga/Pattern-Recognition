from re import S
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class ContentClassifier:
    def __init__(self, y_true, y_pred, num_contents):
        self.y_true = y_true  # 实际请求量，numpy 数组，形状为 [样本数, 内容数]
        self.num_contents = num_contents

    def classify(request_counts, q33, q66):
            categories = []
            for count in request_counts:
                if count <= q33:
                    categories.append("Unpopular")
                elif count <= q66:
                    categories.append("Normal")
                else:
                    categories.append("Popular")
            return categories
    
    def classify_contents(self):
        y_true_last = self.y_true[-1]
        q33, q66 = np.percentile(y_true_last, [33, 66])
        print(f"33% 分位数: {q33:.2f}, 66% 分位数: {q66:.2f}")

        true_categories = classify(y_true_last, q33, q66)

        true_counts = Counter(true_categories)
        print("实际请求量分类统计:")
        print(true_counts)

        labels = ["Unpopular", "Normal", "Popular"]
        sizes_true = [true_counts[label] for label in labels]

        # 实际请求量分类的饼图
        plt.figure(figsize=(8, 6))
        plt.pie(sizes_true, labels=labels, autopct="%1.1f%%", startangle=140)
        plt.title("实际请求量分类")
        plt.axis("equal")
        plt.show()
