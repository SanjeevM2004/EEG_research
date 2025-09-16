import mne
import numpy as np
import matplotlib.pyplot as plt

# Load EEG file
file_path = "EEG_data/Physionet/S001/S001R14.edf"
raw = mne.io.read_raw_edf(file_path, preload=True)

# Pick a single channel
channel = raw.ch_names[0]

# Get raw data for that channel
data = raw.copy().pick_channels([channel]).get_data().flatten()
sfreq = raw.info['sfreq']

# FFT
n = len(data)
freqs = np.fft.rfftfreq(n, 1/sfreq)
fft_vals = np.abs(np.fft.rfft(data))

# PSD using Welch's method
psd = raw.compute_psd(fmin=0.5, fmax=50, n_fft=2048, picks=[channel])
psd_vals, psd_freqs = psd.get_data(return_freqs=True)

# Define frequency bands with colors
bands = {
    'Delta (0.5-4 Hz)':  (0.5, 4,  'lightblue'),
    'Theta (4-8 Hz)':   (4, 8,  'lightgreen'),
    'Alpha (8-13 Hz)':  (8, 13, 'lightyellow'),
    'Beta (13-30 Hz)':  (13, 30,'lightpink'),
    'Gamma (30-50 Hz)': (30, 50,'lavender')
}

# Plot FFT
plt.figure(figsize=(12, 6))
plt.plot(freqs, fft_vals, color='blue', label='FFT Amplitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title(f'Frequency Spectrum of Channel {channel}')

for band, (low, high, color) in bands.items():
    plt.axvspan(low, high, alpha=0.3, color=color, label=band)

plt.xlim(0, 50)
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig("frequency_spectrum_fft.png")
plt.show()

# Plot PSD
plt.figure(figsize=(12, 6))
plt.semilogy(psd_freqs, psd_vals[0], color='green', label='Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (uV^2/Hz)')
plt.title(f'PSD of Channel {channel}')

for band, (low, high, color) in bands.items():
    plt.axvspan(low, high, alpha=0.3, color=color, label=band)

plt.xlim(0, 50)
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig("psd.png")
plt.show()
