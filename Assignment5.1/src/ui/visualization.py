import matplotlib.pyplot as plt

# plot a spectrogram to show what the model looks at
def show_spectrogram(dataset, label_names):
    for spectrogram, label in dataset.take(1): # use first sample
        spec_np = spectrogram[0].numpy().squeeze()  # use spectrogram
        label_val = label[0].numpy() # get label for the example

        plt.figure(figsize=(10, 4)) # create plot
        plt.imshow(spec_np.T, aspect='auto', origin='lower', cmap='magma') 
        plt.title(f"Spectrogram: {label_names[label_val] if label_names else label_val}") # add title
        plt.xlabel("Time") # add x label
        plt.ylabel("Mel Freq") # add y label
        plt.colorbar(format="%+2.0f dB") # add colorbar
        plt.tight_layout() # adjust spacing
        plt.show() # display plot
        break