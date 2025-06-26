import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.dataset import BearingDataset
from model.model import Autoencoder1D
from config import CONFIG

def evaluate():
    test_loader = DataLoader(BearingDataset("data/processed/test.pt"), batch_size=1, shuffle=False)
    device = CONFIG["device"]

    model = Autoencoder1D().to(device)
    model.load_state_dict(torch.load(CONFIG["model_save_path"]))
    model.eval()

    criterion = torch.nn.MSELoss(reduction='none')
    all_losses = []
    all_latencies = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)

            start_time = time.time()
            output = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()

            latency = (end_time - start_time) / x.shape[0]
            all_latencies.extend([latency] * x.shape[0])

            loss = criterion(output, x)
            loss = loss.mean(dim=(1, 2))
            all_losses.extend(loss.cpu().numpy())

    return np.array(all_losses), np.array(all_latencies)

def detect_anomalies(mse_array, method='mean+3std'):
    if method == 'mean+3std':
        threshold = mse_array.mean() + 3 * mse_array.std()
    elif method == '99.5_percentile':
        threshold = np.percentile(mse_array, 99.5)
    else:
        raise ValueError("Unknown method")

    anomalies = np.where(mse_array > threshold)[0]
    return threshold, anomalies

def plot_results(mse_array, threshold, anomalies, latencies):
    plt.figure(figsize=(10, 4))
    plt.hist(mse_array, bins=50, alpha=0.7, color='skyblue')
    plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.2e}")
    plt.title("MSE Distribution")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(mse_array, label="MSE per sample")
    plt.scatter(anomalies, mse_array[anomalies], color='red', label="Anomalies", s=20)
    plt.axhline(threshold, color='gray', linestyle='--')
    plt.title("Reconstruction Error with Anomalies")
    plt.xlabel("Sample index")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Total anomalies: {len(anomalies)}")
    print(f"Avg Latency per sample: {np.mean(latencies) * 1000:.3f} ms")
    print(f"Max Latency: {np.max(latencies) * 1000:.3f} ms")
    print(f"Threshold used: {threshold:.6e}")

if __name__ == '__main__':
    losses, latencies = evaluate()
    threshold, anomalies = detect_anomalies(losses, method='mean+3std')
    plot_results(losses, threshold, anomalies, latencies)
