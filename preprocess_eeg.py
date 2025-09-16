from glob import glob
import mne
from preprocessing.filters import bandpass_filter, notch_filter
from preprocessing.artifacts import run_ica, reject_bad_epochs
from preprocessing.epoching import create_epochs
from preprocessing.normalize import zscore_normalize

file_paths = glob("EEG_data/Physionet/*/*.edf")

for file_path in file_paths[:5]:
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

    # Step 1: Filters
    raw = bandpass_filter(raw, 1, 40)
    raw = notch_filter(raw, 50)

    # Step 2: Artifact removal
    raw = run_ica(raw, n_components=20)

    # Step 3: Epoching
    epochs = create_epochs(raw, tmin=-0.5, tmax=4.0)
    epochs = reject_bad_epochs(epochs)

    # Step 4: Normalization
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    labels = epochs.events[:, -1]
    norm_data = zscore_normalize(data)

    print(f"{file_path}: {norm_data.shape}, labels={set(labels)}")
