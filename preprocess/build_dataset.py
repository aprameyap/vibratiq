### File: preprocess/build_dataset.py
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

WINDOW_SIZE = 1024
STRIDE = 512
SENSOR_IDX = 0  # index of Bearing_1_DE

def load_data_from_dir(path):
    files = sorted(os.listdir(path))
    data = []
    for file in tqdm(files, desc=f"Loading {path}"):
        file_path = os.path.join(path, file)
        raw = pd.read_csv(file_path, sep='\t', header=None)
        signal = raw.iloc[:, SENSOR_IDX].values
        data.append(signal)
    return np.concatenate(data)

def create_windows(data):
    windows = []
    for i in range(0, len(data) - WINDOW_SIZE, STRIDE):
        window = data[i:i+WINDOW_SIZE]
        windows.append(window)
    return np.array(windows)

def preprocess_and_save():
    root = "data"
    all_data = []
    for folder in ["1st_test", "2nd_test/2nd_test", "3rd_test/4th_test/txt"]:
        path = os.path.join(root, folder)
        sig = load_data_from_dir(path)
        all_data.append(sig)
    
    total = np.concatenate(all_data)
    scaler = StandardScaler()
    total = scaler.fit_transform(total.reshape(-1, 1)).flatten()
    
    windows = create_windows(total)
    windows = windows[:, None, :]  # shape (N, 1, WINDOW_SIZE)

    n = len(windows)
    train, val, test = windows[:int(0.7*n)], windows[int(0.7*n):int(0.85*n)], windows[int(0.85*n):]
    os.makedirs("data/processed", exist_ok=True)
    torch.save(train, "data/processed/train.pt")
    torch.save(val, "data/processed/val.pt")
    torch.save(test, "data/processed/test.pt")

if __name__ == '__main__':
    preprocess_and_save()
