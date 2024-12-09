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
        y_true_last = self.y_true[-1]
        y_pred_last = self.y_pred[-1]
        q33, q66 = np.percentile(y_true_last, [33, 66])
        print(f"33% 分位数: {q33:.2f}, 66% 分位数: {q66:.2f}")

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

        true_categories = classify(y_true_last, q33, q66)
        pred_categories = classify(y_pred_last, q33, q66)

        self.true_categories = true_categories
        self.pred_categories = pred_categories

        true_counts = Counter(true_categories)
        pred_counts = Counter(pred_categories)
        print("实际请求量分类统计:")
        print(true_counts)
        print("预测请求量分类统计:")
        print(pred_counts)

        labels = ["Unpopular", "Normal", "Popular"]
        sizes_true = [true_counts[label] for label in labels]
        sizes_pred = [pred_counts[label] for label in labels]

        # 实际请求量分类的饼图
        plt.figure(figsize=(8, 6))
        plt.pie(sizes_true, labels=labels, autopct="%1.1f%%", startangle=140)
        plt.title("实际请求量分类")
        plt.axis("equal")
        plt.show()

        # 预测请求量分类的饼图
        plt.figure(figsize=(8, 6))
        plt.pie(sizes_pred, labels=labels, autopct="%1.1f%%", startangle=140)
        plt.title("预测请求量分类")
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
