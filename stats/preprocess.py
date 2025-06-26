# preprocess_features.py

import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from tqdm import tqdm
from config import DATA_PATH

def extract_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    p2p = np.ptp(signal)
    k = kurtosis(signal)
    crest = np.max(np.abs(signal)) / rms if rms != 0 else 0
    return [rms, p2p, k, crest]

def load_signals():
    files = sorted(os.listdir(DATA_PATH))
    feature_list = []

    for f in tqdm(files, desc='Extracting'):
        path = os.path.join(DATA_PATH, f)
        sig = np.loadtxt(path, usecols=0)  # Assume col 0 is Bearing 1 DE
        feat = extract_features(sig)
        feature_list.append(feat)

    return np.array(feature_list), files

if __name__ == '__main__':
    features, files = load_signals()
    df = pd.DataFrame(features, columns=['rms', 'peak_to_peak', 'kurtosis', 'crest_factor'])
    df['filename'] = files
    df.to_csv('features.csv', index=False)
