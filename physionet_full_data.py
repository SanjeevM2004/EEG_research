from glob import glob 
import mne

# Define the file path to the EDF file
file_paths = glob("EEG_data/Physionet/*/*.edf")

for file_path in file_paths:
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose = False)
    events, event_id = mne.events_from_annotations(raw, verbose = False)
    tmin, tmax = -0.5, 4
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True, verbose = False)
    temp_data = epochs.get_data()
    labels = epochs.events[:, -1]
    print(f"file: {file_path[19:23]}, Data shape: {temp_data.shape}, Labels shape: {labels.shape}, Unique labels: {set(labels)}")

