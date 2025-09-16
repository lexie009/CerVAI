import os

config_folder = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "data_config": os.path.join(config_folder, "data_config.yaml"),
    "model_config": os.path.join(config_folder, "model_config.yaml"),
    "train_config": os.path.join(config_folder, "train_config.yaml"),
    "sampling_config": os.path.join(config_folder, "sampling/BordaBatch.yaml"),
    "logger_config": os.path.join(config_folder, "logger_config.yaml")
}

