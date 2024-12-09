import os
import yaml
import argparse


class Config:
    """
    Config 类，用于加载配置文件和解析命令行参数。

    Args:
        config_file (str): 配置文件路径，默认为 "config.yaml"
    """

    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        # 设置默认参数
        self.num_contents = 1000
        self.time_steps = 500
        self.popular_size = 50
        self.sequence_length = 10
        self.batch_size = 32
        self.hidden_size = 64
        self.num_layers = 1
        self.dropout = 0.0
        self.learning_rate = 0.001
        self.num_epochs = 50
        self.patience = 5  # 早停的容忍度
        self.min_delta = 0.0001  # 最小改进量
        self.log_level = "INFO"
        self.raw_data_path = "data/raw_data.csv"
        self.best_model_path = "models/best_model.pth"
        # 训练集, 验证集和测试集的比例
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 1 - self.train_ratio - self.val_ratio
        self.num_linear_layers = 1

        # 检查配置文件是否存在
        if not os.path.exists(self.config_file):
            print(f"配置文件 '{self.config_file}' 未找到，正在创建默认配置文件。")
            self.create_default_config()
        else:
            self.load_config()

        # 解析命令行参数
        self.parse_args()

    def create_default_config(self):
        # 创建默认配置字典
        default_config = {
            "num_contents": self.num_contents,
            "time_steps": self.time_steps,
            "popular_size": self.popular_size,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "log_level": self.log_level,
            "raw_data_path": self.raw_data_path,
            "best_model_path": self.best_model_path,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "num_linear_layers": self.num_linear_layers,
        }
        # 将默认配置写入配置文件
        with open(self.config_file, "w") as f:
            yaml.dump(default_config, f)
        print(f"默认配置文件 '{self.config_file}' 已创建。")

    def load_config(self):
        # 加载配置文件
        with open(self.config_file, "r") as f:
            config = yaml.safe_load(f)
            self.num_contents = config.get("num_contents", self.num_contents)
            self.time_steps = config.get("time_steps", self.time_steps)
            self.popular_size = config.get("popular_size", self.popular_size)
            self.sequence_length = config.get("sequence_length", self.sequence_length)
            self.batch_size = config.get("batch_size", self.batch_size)
            self.hidden_size = config.get("hidden_size", self.hidden_size)
            self.num_layers = config.get("num_layers", self.num_layers)
            self.dropout = config.get("dropout", self.dropout)
            self.learning_rate = config.get("learning_rate", self.learning_rate)
            self.num_epochs = config.get("num_epochs", self.num_epochs)
            self.log_level = config.get("log_level", self.log_level)
            self.patience = config.get("patience", self.patience)
            self.min_delta = config.get("min_delta", self.min_delta)
            self.raw_data_path = config.get("raw_data_path", self.raw_data_path)
            self.best_model_path = config.get("best_model_path", self.best_model_path)
            self.val_ratio = config.get("val_ratio", self.val_ratio)
            self.test_ratio = config.get("test_ratio", self.test_ratio)
            self.log_level = config.get("log_level", self.log_level)
            self.num_linear_layers = config.get(
                "num_linear_layers", self.num_linear_layers
            )

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="CDN Content Popularity Prediction"
        )
        parser.add_argument("--num_contents", type=int, help="Number of contents")
        parser.add_argument("--time_steps", type=int, help="Number of time steps")
        parser.add_argument(
            "--popular_size", type=int, help="Number of popular contents"
        )
        parser.add_argument(
            "--sequence_length", type=int, help="Sequence length for LSTM"
        )
        parser.add_argument("--batch_size", type=int, help="Batch size for training")
        parser.add_argument("--hidden_size", type=int, help="Hidden size of LSTM")
        parser.add_argument("--num_layers", type=int, help="Number of LSTM layers")
        parser.add_argument("--dropout", type=float, help="Dropout rate")
        parser.add_argument("--learning_rate", type=float, help="Learning rate")
        parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
        parser.add_argument("--log_level", type=str, help="Logging level")
        parser.add_argument("--patience", type=int, help="Early stopping patience")
        parser.add_argument(
            "--min_delta",
            type=float,
            help="Minimum change in validation loss to qualify as improvement",
        )
        parser.add_argument(
            "--raw_data_path", type=str, help="Path to save raw data CSV file"
        )
        parser.add_argument(
            "--best_model_path", type=str, help="Path to save the best model"
        )
        parser.add_argument("--train_ratio", type=float, help="Training set ratio")
        parser.add_argument("--val_ratio", type=float, help="Validation set ratio")
        parser.add_argument("--test_ratio", type=float, help="Test set ratio")
        parser.add_argument(
            "--num_linear_layers", type=int, help="Number of linear layers"
        )

        args = parser.parse_args()

        # 如果命令行提供了参数，则覆盖配置文件中的值
        if args.num_contents is not None:
            self.num_contents = args.num_contents
        if args.time_steps is not None:
            self.time_steps = args.time_steps
        if args.popular_size is not None:
            self.popular_size = args.popular_size
        if args.sequence_length is not None:
            self.sequence_length = args.sequence_length
        if args.batch_size is not None:
            self.batch_size = args.batch_size
        if args.hidden_size is not None:
            self.hidden_size = args.hidden_size
        if args.num_layers is not None:
            self.num_layers = args.num_layers
        if args.dropout is not None:
            self.dropout = args.dropout
        if args.learning_rate is not None:
            self.learning_rate = args.learning_rate
        if args.num_epochs is not None:
            self.num_epochs = args.num_epochs
        if args.log_level is not None:
            self.log_level = args.log_level.upper()
        if args.patience is not None:
            self.patience = args.patience
        if args.min_delta is not None:
            self.min_delta = args.min_delta
        if args.train_ratio is not None:
            self.train_ratio = args.train_ratio
        if args.val_ratio is not None:
            self.val_ratio = args.val_ratio
        if args.test_ratio is not None:
            self.test_ratio = args.test_ratio
