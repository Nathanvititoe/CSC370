import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import Audio, display, Markdown
import matplotlib.lines as mlines

# get jupyter to display plots in-line
try:
    get_ipython() # type: ignore
except NameError:
    import matplotlib
    matplotlib.use('TkAgg')

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

# visualize loss v acc during training and validation 
def visualize_stats(classifier_history):
    # get metrics
    train_loss = classifier_history.history.get('loss', [])
    val_loss = classifier_history.history.get('val_loss', [])
    train_acc = classifier_history.history.get('accuracy', [])
    val_acc = classifier_history.history.get('val_accuracy', [])

    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12,5)) # set plot size

    # Plot the training vs. validation Accuracy
    plt.subplot(1,2,1) 
    plt.plot(epochs, train_acc, label='Training Accuracy') # Training acc plot
    plt.plot(epochs, val_acc, label='Validation Accuracy') # Validation acc plot
    plt.title('Training vs Validation Accuracy') # Title
    plt.xlabel('Epoch') # x label
    plt.ylabel('Accuracy') # y label
    plt.legend() # legend

    # plot training v. validation Loss
    plt.subplot(1,2,2) 
    plt.plot(epochs, train_loss, label='Training Loss') # training loss plot
    plt.plot(epochs, val_loss, label='Validation Loss') # validation loss plot
    plt.title('Training vs Validation Loss') # title
    plt.xlabel('Epoch') # x label
    plt.ylabel('Loss') # y label
    plt.legend() # legend

    plt.tight_layout() # adjust spacing between the plots
    plt.show() # display them

#TODO: comment and clean up
# function to plot/compare raw v processed audio waveforms
def plot_waveform_comparison(class_waveforms_raw, class_waveforms_proc):
    num_classes = len(class_waveforms_raw) # get number of classes
    fig, axes = plt.subplots(num_classes, 2, figsize=(10, 2 * num_classes), constrained_layout=True) # create subplots
    fig.suptitle("Raw vs. Processed Audio Waveforms\n\n", fontsize=14) # add title (position and fontsize)
    
    # iterate through audio waveform dict (for each class) 
    for i, label in enumerate(class_waveforms_raw):
        raw_waveform, sr = class_waveforms_raw[label] # get raw waveform
        proc_waveform, sr = class_waveforms_proc[label] # get processed waveform

        # convert samples to time for human readability (x axis)
        raw_time = np.arange(len(raw_waveform)) / sr 
        proc_time = np.arange(len(proc_waveform)) / sr

        # plot raw audio waveform channels
        if raw_waveform.ndim == 2:  # if stereo audio
            axes[i, 0].plot(raw_time, raw_waveform[:, 0], label='Left', color='blue', alpha=0.5) # plot left channel 
            axes[i, 0].plot(raw_time, raw_waveform[:, 1], label='Right', color='red', alpha=0.5) # plot right channel
            axes[i, 0].set_xlim(raw_time[0], raw_time[-1]) # trim stereo whitespace
        else:  # if mono audio
            axes[i, 0].plot(raw_time, raw_waveform, label='Mono', color='blue') # plot single channel
            axes[i, 0].set_xlim(raw_time[0], raw_time[-1]) # trim mono whitespace
        
        # Raw Audio Titles
        axes[i, 0].set_title(f"{label} - Raw", fontsize=9) # add title for each subplot
        axes[i, 0].set_ylabel("Amplitude", fontsize=7) # add y label

        # processed audio plot and title
        axes[i, 1].plot(proc_time, proc_waveform, color='purple') # create plot
        axes[i, 1].set_title(f"{label} - Processed", fontsize=9) # add title
        axes[i, 1].set_xlim(proc_time[0], proc_time[-1])  # trim processed whitespace

    # add x label
    for ax in axes.flat:
        ax.set_xlabel("Time (sec)", fontsize=7) # add x labels
        ax.tick_params(axis='both', labelsize=6, length=2, pad=2) # set sizing for axis markings

    # create legend to show mono v. stereo
    legend_lines = [
    mlines.Line2D([], [], color='blue', label='Left Channel'),
    mlines.Line2D([], [], color='red', label='Right Channel'),
    mlines.Line2D([], [], color='purple', label='Processed Mono')
    ]

    fig.legend(handles=legend_lines, loc='upper left', ncol=3, fontsize='medium') # add legend
    plt.tight_layout() # adjust layout
    # plt.tight_layout(rect=[0, 0, 1, 0.98])  # leaves ~8% space for title/legend
    plt.show() # display plots

# display one spectrogram per class
def plot_spectrograms(class_spectrograms):
    labels = sorted(class_spectrograms.keys()) # get class labels
    num_classes = len(labels) # get total number of classes
    num_cols = 4 # number of columns
    num_rows = int(np.ceil(num_classes / num_cols)) # number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows)) # create subplots for each class
    axes = axes.flatten() # allow for easy iteration over axes

    # iterate through class_spectrograms
    for i, label in enumerate(labels):
        spec = class_spectrograms[label] # get the spectrogram
        ax = axes[i] # select subplot
        ax.imshow(spec, aspect='auto', origin='lower', cmap='magma') # show the spectrogram
        ax.set_title(f"{label}", fontsize=10) # add title (label)
        ax.axis('off') # turn off axis labels

    # disable unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle("UrbanSound Spectrograms", fontsize=16) # add title
    plt.tight_layout() # adjust layout
    plt.show() # display plot

# function to display an audio sample to the user
def audio_sampler(filepath, sample_rate, duration, label):
    from src.prep_data.preprocess import load_file # import here to avoid cyclic error
    try:
        num_samples = sample_rate * duration # get num_samples
        audio = load_file(filepath, sample_rate, num_samples).numpy() # load sample file
 
        display(Markdown(f"**{label}**")) # add title
        display(Audio(audio, rate=sample_rate)) # show audio sample

    except Exception as e:
        print(f"Could not load {filepath}:\n→ {e}")


# function to display confusion matrix and monitor what the model is struggling with
def plot_confusion_matrix(audio_classifier, val_features, val_labels, label_names):
    y_pred = audio_classifier.predict(val_features) # get prediction on validation features
    y_pred_labels = np.argmax(y_pred, axis=1) # get prediction labels

    fig, ax = plt.subplots(figsize=(7.5, 8.5))  # adjust size to fit everything

    # create scikit confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
            val_labels, y_pred_labels,
            display_labels=label_names,
            cmap='viridis',
            ax=ax
            )
    # add title/subtitle
    disp.ax_.set_title("Confusion Matrix", fontsize=14)
    # disp.ax_.text(
    #     0.5, 0.955,  # x and y in axis coordinates
    #     "Correct predictions per actual class",
    #     fontsize=10,
    #     ha='center',
    #     transform=disp.ax_.transAxes
    # )

    # rotate x axis labels for readability
    disp.ax_.set_xticklabels(disp.ax_.get_xticklabels(), rotation=45, ha='right')

    # plt.subplots_adjust(top=0.88, bottom=0.15)
    plt.tight_layout() # adjust layout
    plt.show() # display plot