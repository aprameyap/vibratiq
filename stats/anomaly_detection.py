# anomaly_detection.py

import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from config import ROLLING_WINDOW, ANOMALY_THRESHOLD

def compute_mahalanobis(X, window=ROLLING_WINDOW):
    anomalies = []
    scores = []

    for i in range(window, len(X)):
        ref = X[i-window:i]
        cov = np.cov(ref, rowvar=False)
        if np.linalg.det(cov) == 0:
            cov += np.eye(cov.shape[0]) * 1e-6

        inv_cov = np.linalg.inv(cov)
        mu = np.mean(ref, axis=0)
        dist = mahalanobis(X[i], mu, inv_cov)
        scores.append(dist)
        anomalies.append(dist > ANOMALY_THRESHOLD)

    return np.array(scores), np.array(anomalies)

if __name__ == '__main__':
    df = pd.read_csv('features.csv')
    X = df[['rms', 'peak_to_peak', 'kurtosis', 'crest_factor']].values
    scores, anomalies = compute_mahalanobis(X)

    df = df.iloc[ROLLING_WINDOW:].copy()
    df['mahalanobis'] = scores
    df['anomaly'] = anomalies
    df.to_csv('anomaly_output.csv', index=False)
