import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import find_peaks

# Define the features to extract
FEATURES = {
    "time_domain": [
        "mean", "std", "aad", "min", "max", "maxmin_diff", "median",
        "mad", "IQR", "neg_count", "pos_count", "above_mean",
        "peak_count", "skewness", "kurtosis", "energy"
    ],
    "freq_domain": [
        "mean", "std", "aad", "min", "max", "maxmin_diff", "median",
        "mad", "IQR", "above_mean", "peak_count", "skewness",
        "kurtosis", "energy"
    ]
}

# Define feature extraction function
def extract_features(signal, features):
    """Extracts selected statistical features from a signal."""
    stats_funcs = {
        "mean": lambda x: x.mean(),
        "std": lambda x: x.std(),
        "aad": lambda x: np.mean(np.abs(x - np.mean(x))),
        "min": lambda x: x.min(),
        "max": lambda x: x.max(),
        "maxmin_diff": lambda x: x.max() - x.min(),
        "median": lambda x: np.median(x),
        "mad": lambda x: np.median(np.abs(x - np.median(x))),
        "IQR": lambda x: np.percentile(x, 75) - np.percentile(x, 25),
        "neg_count": lambda x: np.sum(x < 0),
        "pos_count": lambda x: np.sum(x > 0),
        "above_mean": lambda x: np.sum(x > x.mean()),
        "peak_count": lambda x: len(find_peaks(x)[0]),
        "skewness": lambda x: stats.skew(x),
        "kurtosis": lambda x: stats.kurtosis(x),
        "energy": lambda x: np.sum(x**2) / len(x)
    }
    
    return {feature: stats_funcs[feature](signal) for feature in features}

# Load the dataset
df_train = pd.read_csv("data.csv")  # Replace with actual filename

# Define windowing parameters
window_size = 100
step_size = 50

# Create DataFrame to store extracted features
X_train = pd.DataFrame()

# Sliding window approach
for i in range(0, df_train.shape[0] - window_size, step_size):
    window_x = df_train['X'].values[i: i + window_size]
    window_y = df_train['Y'].values[i: i + window_size]
    window_z = df_train['Z'].values[i: i + window_size]

    # Extract time-domain features
    features_x = extract_features(window_x, FEATURES["time_domain"])
    features_y = extract_features(window_y, FEATURES["time_domain"])
    features_z = extract_features(window_z, FEATURES["time_domain"])

    # Compute FFT (frequency-domain features)
    fft_x = np.abs(np.fft.fft(window_x))[1:window_size // 2 + 1]
    fft_y = np.abs(np.fft.fft(window_y))[1:window_size // 2 + 1]
    fft_z = np.abs(np.fft.fft(window_z))[1:window_size // 2 + 1]

    features_x_fft = extract_features(fft_x, FEATURES["freq_domain"])
    features_y_fft = extract_features(fft_y, FEATURES["freq_domain"])
    features_z_fft = extract_features(fft_z, FEATURES["freq_domain"])

    # Combine features into a row
    row = {f"x_{k}": v for k, v in features_x.items()}
    row.update({f"y_{k}": v for k, v in features_y.items()})
    row.update({f"z_{k}": v for k, v in features_z.items()})
    row.update({f"x_fft_{k}": v for k, v in features_x_fft.items()})
    row.update({f"y_fft_{k}": v for k, v in features_y_fft.items()})
    row.update({f"z_fft_{k}": v for k, v in features_z_fft.items()})

    # Assign label (1 if tap detected in window, else 0)
    row["Label"] = 1 if 1 in df_train['Tap'][i: i + window_size].values else 0

    # Append row to DataFrame
    X_train = X_train.append(row, ignore_index=True)

# Save to CSV
X_train.to_csv("X_train.csv", index=False)

print("Feature extraction complete. Saved as X_train.csv")
