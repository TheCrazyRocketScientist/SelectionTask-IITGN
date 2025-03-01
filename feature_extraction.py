import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks







def extract_features(window):
    # Split into separate axes
    x_window, y_window, z_window = window[:, 0], window[:, 1], window[:, 2]

    features = {}
    
    # Time domain features
    features['z_mean'] = np.mean(z_window)
    features['z_std'] = np.std(z_window)
    features['z_aad'] = np.mean(np.abs(z_window - np.mean(z_window)))
    
    #features['z_max'] = np.max(z_window)
    #features['z_min'] = np.min(z_window)
    features['z_maxmin_diff'] = np.max(z_window) - np.min(z_window)
    
    #features['z_peak_count'] = len(find_peaks(z_window)[0])
    features['z_skewness'] = stats.skew(z_window)
    features['z_kurtosis'] = stats.kurtosis(z_window)
    features['z_energy'] = np.sum(z_window**2 / len(z_window))
    features['z_above_mean'] = np.sum(z_window > np.mean(z_window))  # Fixed here
    
    # Magnitude and SMA
    resultant_accl = np.sqrt(x_window**2 + y_window**2 + z_window**2)
    features['avg_result_accl'] = np.mean(resultant_accl)
    features['sma'] = np.sum(np.abs(x_window) + np.abs(y_window) + np.abs(z_window)) / len(z_window)
    
    # Frequency domain features (FFT)
    z_fft = np.abs(np.fft.fft(z_window))[1:len(z_window)//2+1]
    
    features['z_mean_fft'] = np.mean(z_fft)
    features['z_std_fft'] = np.std(z_fft)
    features['z_aad_fft'] = np.mean(np.abs(z_fft - np.mean(z_fft)))
    
    features['z_max_fft'] = np.max(z_fft)
    features['z_min_fft'] = np.min(z_fft)
    features['z_maxmin_diff_fft'] = features['z_max_fft'] - features['z_min_fft']
    
    features['z_median_fft'] = np.median(z_fft)
    features['z_mad_fft'] = np.median(np.abs(z_fft - np.median(z_fft)))
    features['z_IQR_fft'] = np.percentile(z_fft, 75) - np.percentile(z_fft, 25)
    
    features['z_above_mean_fft'] = np.sum(z_fft > np.mean(z_fft))
    features['z_skewness_fft'] = stats.skew(z_fft)
    features['z_kurtosis_fft'] = stats.kurtosis(z_fft)
    features['z_energy_fft'] = np.sum(z_fft**2 / len(z_fft))
    features['z_peak_count_fft'] = len(find_peaks(z_fft)[0])  # Fixed here
    
    resultant_accl_fft = np.sqrt(
        np.abs(np.fft.fft(x_window))**2 + 
        np.abs(np.fft.fft(y_window))**2 + 
        np.abs(np.fft.fft(z_window))**2
    )
    features['avg_result_accl_fft'] = np.mean(resultant_accl_fft[1:len(z_window)//2+1])

    return features
