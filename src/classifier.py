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
        self.y_pred = y_pred  # 预测请求量，numpy 数组，形状为 [样本数, 内容数]
        self.num_contents = num_contents

    def classify_contents(self):
        # 对最后一个时间步的数据进行分类
        y_true_last = self.y_true[-1]
        y_pred_last = self.y_pred[-1]

        # 定义实际请求量的阈值
        q33_true, q66_true = np.percentile(y_true_last, [33, 66])

        # 对实际请求量进行分类
        true_categories = []
        for count in y_true_last:
            if count <= q33_true:
                true_categories.append("Unpopular")
            elif count <= q66_true:
                true_categories.append("Normal")
            else:
                true_categories.append("Popular")
        self.true_categories = true_categories

        # 对预测请求量进行分类，使用相同的阈值
        pred_categories = []
        for count in y_pred_last:
            if count <= q33_true:
                pred_categories.append("Unpopular")
            elif count <= q66_true:
                pred_categories.append("Normal")
            else:
                pred_categories.append("Popular")
        self.pred_categories = pred_categories

        # 统计实际分类结果
        true_counts = Counter(true_categories)
        print(f"实际请求量分类统计:")
        print(true_counts)

        # 统计预测分类结果
        pred_counts = Counter(pred_categories)
        print(f"预测请求量分类统计:")
        print(pred_counts)

        # 绘制实际请求量分类的饼图
        labels = ["Unpopular", "Normal", "Popular"]
        sizes_true = [true_counts.get(label, 0) for label in labels]
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.pie(sizes_true, labels=labels, autopct="%1.1f%%", startangle=140)
        plt.title(f"实际请求量分类")

        # 绘制预测请求量分类的饼图
        sizes_pred = [pred_counts.get(label, 0) for label in labels]
        plt.subplot(1, 2, 2)
        plt.pie(sizes_pred, labels=labels, autopct="%1.1f%%", startangle=140)
        plt.title(f"预测请求量分类")

        plt.axis("equal")
        plt.show()

    def evaluate_classification(self):
        # 将类别映射为数字
        label_mapping = {"Unpopular": 0, "Normal": 1, "Popular": 2}
        y_true_labels = np.array(
            [label_mapping[label] for label in self.true_categories]
        )
        y_pred_labels = np.array(
            [label_mapping[label] for label in self.pred_categories]
        )

        # 计算分类报告
        from sklearn.metrics import classification_report, confusion_matrix

        print("分类报告:")
        print(
            classification_report(
                y_true_labels,
                y_pred_labels,
                target_names=["Unpopular", "Normal", "Popular"],
            )
        )

        # 绘制混淆矩阵
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Unpopular", "Normal", "Popular"],
            yticklabels=["Unpopular", "Normal", "Popular"],
        )
        plt.xlabel("预测类别")
        plt.ylabel("实际类别")
        plt.title("混淆矩阵")
        plt.show()
