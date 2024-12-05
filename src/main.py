from config import Config
from data import DataGenerator
from lstm import LSTMModel
from train import ModelTrainer
from visualize import Visualizer


def main():
    # 加载配置
    config = Config()

    # 数据生成和预处理
    data_gen = DataGenerator(config)
    data_gen.generate_data()
    data_gen.preprocess_data()
    train_loader, val_loader, _ = data_gen.get_data_loaders(
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )

    # 模型定义
    model = LSTMModel(config)

    # 模型训练
    trainer = ModelTrainer(model, train_loader, val_loader, config)
    trainer.train()
    print("模型训练完成")

    # 模型测试
    test_X = data_gen.X[data_gen.test_indices]
    test_y = data_gen.y[data_gen.test_indices]
    y_true, y_pred = trainer.evaluate(test_X, test_y, data_gen.scaler)
    print("测试集评估完成")

    # 可视化和分析
    visualizer = Visualizer(y_true, y_pred, config.num_contents)
    visualizer.plot_loss(trainer.train_losses, trainer.val_losses)
    visualizer.plot_predictions(num_samples=5)
    visualizer.plot_distribution()
    visualizer.plot_top_n(top_n=10)
    visualizer.classify_contents()


if __name__ == "__main__":
    main()
