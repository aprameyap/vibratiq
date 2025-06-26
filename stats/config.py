# config.py

SAMPLE_LENGTH = 20480  # per file
FEATURES = ['rms', 'peak_to_peak', 'kurtosis', 'crest_factor']
ROLLING_WINDOW = 100  # number of past points for reference
ANOMALY_THRESHOLD = 4.5  # Mahalanobis threshold (adjustable)
DATA_PATH = 'data/1st_test'
