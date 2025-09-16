import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The file path for the EEG data
file_path = "EEG_data/Physionet/S001/S001R14.edf"

# Load the raw EEG data and get events
raw = mne.io.read_raw_edf(file_path, preload=True)
events, event_id = mne.events_from_annotations(raw)
tmin, tmax = -2, 4

# Create epochs from the raw data based on events
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True)

# Get the unique labels to create separate plots for each event type
unique_labels = sorted(list(event_id.keys()))

# Loop through each unique label to create a plot
for label in unique_labels:
    # Select epochs for the current label
    epochs_for_label = epochs[label]

    if len(epochs_for_label) > 0:
        # Get the 3D data array (n_epochs, n_channels, n_times)
        data = epochs_for_label.get_data()

        # Get the integer labels for the current epochs
        labels = epochs_for_label.events[:, -1]

        # Reshape the data to a 2D format (n_epochs, n_channels * n_times)
        n_epochs = data.shape[0]
        print(data.shape)
        flattened_data = data.reshape(n_epochs, -1)

        # Create a DataFrame from the flattened data
        df = pd.DataFrame(flattened_data)
        # Check if any epochs exist for the label

        # Add the labels as the last column
        df['label'] = labels
        print(df.shape)
        # Get the time points for the x-axis
        times = epochs_for_label.times
        
        # Create a new figure and a single axis for the plot
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot each individual epoch's mean (averaged across channels)
        for i in range(data.shape[0]):
            single_epoch_mean = np.mean(data[i, :, :], axis=0)
            ax.plot(times, single_epoch_mean*10**6, color='gray', linestyle='-', alpha=0.5)
        
        # Calculate the overall average ERP
        evoked_data = epochs_for_label.average().get_data()
        print(evoked_data.shape)
        # Calculate the mean across channels for the evoked data
        evoked_mean = np.mean(evoked_data, axis=0)
        evoked_std = np.std(evoked_data, axis=0)

        # Plot the average ERP as a bold line on top of the individual plots
        ax.plot(times, evoked_mean*10**6, color='blue', linewidth=2.5, label='Average ERP')
        ax.fill_between(times, (evoked_mean - evoked_std)*10**6, (evoked_mean + evoked_std)*10**6, color='orange', alpha=1, label='±1 Std Dev')
        
        # Add titles and labels for clarity
        ax.set_title(f"Individual Epochs and Average ERP for Event '{label}'")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (uV)")
        ax.axvline(0, color='r', linestyle='--', label='Event onset')
        ax.grid(True)
        
        # Create a legend
        ax.plot([], [], color='gray', linewidth=0.5, label='Individual Epochs')
        ax.plot([], [], color='blue', linewidth=2.5, label='Average ERP')
        ax.legend()
        
        # Adjust the layout and save the figure
        plt.tight_layout()
        plt.savefig(f"individual_and_average_erp_{label}.png")
        
        # Close the figure to free up memory
        plt.close(fig)
print("All plots generated successfully.")