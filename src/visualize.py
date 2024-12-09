from collections import Counter
from matplotlib import pyplot as plt
import numpy as np

from classifier import ContentClassifier

# 设置全局字体为 SimHei（黑体）
plt.rcParams["font.sans-serif"] = ["SimHei"]


class Visualizer:
    """
    Visualizer 类用于可视化和分析模型的训练和预测结果。
    """

    def __init__(self, y_true, y_pred, num_contents):
        self.y_true = y_true
        self.y_pred = y_pred
        self.num_contents = num_contents

    def plot_loss(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()

    def plot_predictions(self, num_samples=5):
        sample_indices = np.random.choice(self.num_contents, num_samples, replace=False)
        y_true_samples = self.y_true[:, sample_indices]
        y_pred_samples = self.y_pred[:, sample_indices]
        time_axis = range(len(y_true_samples))
        plt.figure(figsize=(12, 8))
        for i in range(num_samples):
            plt.subplot(num_samples, 1, i + 1)
            plt.plot(time_axis, y_true_samples[:, i], label="Actual")
            plt.plot(time_axis, y_pred_samples[:, i], label="Predicted")
            plt.title(f"Content ID: {sample_indices[i]}")
            plt.xlabel("Time Step")
            plt.ylabel("Request Count")
            plt.legend()
            plt.tight_layout()
        plt.show()

    def plot_distribution(self):
        y_true_last = self.y_true[-1]
        y_pred_last = self.y_pred[-1]
        plt.figure(figsize=(12, 6))
        plt.hist(y_true_last, bins=50, alpha=0.5, label="Actual")
        plt.hist(y_pred_last, bins=50, alpha=0.5, label="Predicted")
        plt.xlabel("Request Count")
        plt.ylabel("Frequency")
        plt.title(
            "Distribution of Actual and Predicted Request Counts at Last Time Step"
        )
        plt.legend()
        plt.show()

    def plot_top_n(self, top_n=10):
        y_true_last = self.y_true[-1]
        y_pred_last = self.y_pred[-1]
        top_indices = np.argsort(-y_pred_last)[:top_n]
        top_y_true = y_true_last[top_indices]
        top_y_pred = y_pred_last[top_indices]
        plt.figure(figsize=(12, 6))
        indices = np.arange(top_n)
        width = 0.35
        plt.bar(indices, top_y_true, width=width, label="Actual")
        plt.bar(indices + width, top_y_pred, width=width, label="Predicted")
        plt.xlabel("Content Index")
        plt.ylabel("Request Count")
        plt.title(f"Top {top_n} Predicted Popular Contents")
        plt.xticks(indices + width / 2, top_indices)
        plt.legend()
        plt.show()

    def classify_contents(self):
        # 使用 ContentClassifier 进行分类
        classifier = ContentClassifier(self.y_true, self.y_pred, self.num_contents)
        classifier.classify_contents()
        classifier.evaluate_classification()
