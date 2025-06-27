import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from model.model import Autoencoder1D
from config import CONFIG

signal = torch.load("data/processed/test.pt", weights_only=False)

model = Autoencoder1D().to(CONFIG["device"])
model.load_state_dict(torch.load(CONFIG["model_save_path"]))
model.eval()

threshold = 0.00015
window_delay = 0.01  # 100ms delay to simulate streaming

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(np.zeros(1024))
ax.set_ylim(-4, 4)

print("Real-time streaming started...\n")

latencies = []

for i, sample in enumerate(signal):
    sample = torch.tensor(sample).unsqueeze(0).float().to(CONFIG["device"])

    start_time = time.perf_counter()
    with torch.no_grad():
        output = model(sample)
        mse = torch.nn.functional.mse_loss(output, sample, reduction="mean").item()
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    latencies.append(latency_ms)

    is_anomaly = mse > threshold
    status = "Anomaly Detected" if is_anomaly else "Normal"
    if is_anomaly:
        line.set_color('red')
    else:
        line.set_color('blue')
    line.set_ydata(sample.cpu().numpy()[0][0])
    ax.set_title(f"Sample {i} | MSE: {mse:.2e} | {status} | Latency: {latency_ms:.2f} ms")
    fig.canvas.draw()
    fig.canvas.flush_events()

    print(f"[{i}] MSE: {mse:.6f} | Latency: {latency_ms:.2f} ms → {status}")

    if (i + 1) % 50 == 0:
        avg_latency = np.mean(latencies)
        max_latency = np.max(latencies)
        print(f"\n[Stats] Processed {i + 1} samples | Avg Latency: {avg_latency:.3f} ms | Max: {max_latency:.3f} ms\n")

    time.sleep(window_delay)
