# visualize.py

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('anomaly_output.csv')

plt.figure(figsize=(10, 4))
plt.plot(df['mahalanobis'], label='Mahalanobis Score')
plt.axhline(3.0, color='r', linestyle='--', label='Threshold')
plt.scatter(df.index[df['anomaly']], df['mahalanobis'][df['anomaly']], color='red', label='Anomalies')
plt.title('Anomaly Detection')
plt.xlabel('Time Index')
plt.ylabel('Distance')
plt.legend()
plt.tight_layout()
plt.show()
