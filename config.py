import torch

CONFIG = {
    "batch_size": 128,
    "epochs": 30,
    "learning_rate": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_path": "model/best_model.pt",
    "log_interval": 1,
}