import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def butterworth_filter(data, cutoff=5, fs=50, order=4, filter_type='high'):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data)

def plot_filtered_graph(file_path, column='Z', cutoff=5, fs=50, order=4):
    df = pd.read_csv(file_path)
    if column in df.columns:
        
        plt.figure(figsize=(12, 5))
        plt.plot(df[column], linestyle='-', label=f'Original {column}')
        plt.title(f'Original {column}')
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print(f"Column '{column}' not found in the CSV file.")

if __name__ == "__main__":
    file_path = "imu_data.csv"
    plot_filtered_graph(file_path, column='Z', cutoff=5, fs=50, order=4)
    import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv(file_path)  # Change to your actual file

# Choose the column (e.g., 'Z' for accelerometer data)
column_name = 'X'
if column_name in df.columns:
    signal = df[column_name].values
    n = len(signal)
    
    # Compute FFT
    fft_values = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(n, d=1/100)  # Assuming 100 Hz sampling rate

    # Plot FFT magnitude
    plt.figure(figsize=(12, 5))
    plt.plot(fft_freqs[:n//2], np.abs(fft_values[:n//2]))  # Plot only positive frequencies
    plt.title(f"FFT of {column_name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()
else:
    print(f"Column '{column_name}' not found in the dataset.")

