import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Analyze Audio File Durations (4 subplots)
def visualize_audio_durations(durations, labels, label_names):
    # create pandas dataframe with durations and class labels
    df = pd.DataFrame({
        "duration_sec": durations,
        "classID": labels
    })

    # use label strings for graphs
    df["class_name"] = df["classID"].apply(lambda i: label_names[i] if i < len(label_names) else f"Class {i}")

    # TODO: adjust the outlier recognition depending on avg duration
    # highlight any file with > 1s duration
    df["is_outlier"] = df["duration_sec"] < 1.0

    # create 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(16, 10)) # create plot w subplots
    fig.suptitle("Audio File Duration", fontsize=16) # set subtitle

    # create a histogram for file durations
    sns.histplot(durations, bins=30, kde=True, ax=axs[0, 0]) # create histogram
    axs[0, 0].set_title("Audio File Duration") # set title
    axs[0, 0].set_xlabel("Duration (seconds)") # set x label
    axs[0, 0].set_ylabel("Count") # set y label
    axs[0, 0].grid(True) # turn grid overlay on

    # create a boxplot of file durations
    sns.boxplot(x=durations, ax=axs[0, 1])
    axs[0, 1].set_title("Audio File Duration") # set title
    axs[0, 1].set_xlabel("Duration (seconds)") # set x label
    axs[0, 1].grid(True) # turn grid overlay on

    # group audio file durations by class ID, create a boxplot
    sns.boxplot(x="class_name", y="duration_sec", data=df, ax=axs[1, 0]) # create boxplot
    axs[1, 0].set_title("Audio Duration by Class") # set title
    axs[1, 0].set_xlabel("Class") # set x label
    axs[1, 0].set_ylabel("Duration (seconds)") # set y label
    axs[1, 0].grid(True) # turn grid overlay on

    # create a scatter plot to highlight outliers
    axs[1, 1].scatter(
        df.index,  # x axis: file index
        df["duration_sec"],  # y axis: file duration
        c=df["is_outlier"].map({True: 'red', False: 'black'}),  # make outliers red
        alpha=0.7  # adjust opacity
    )
    axs[1, 1].set_title("Duration Outliers (Red = < 1 sec)") # set title
    axs[1, 1].set_xlabel("File Index") # set x label
    axs[1, 1].set_ylabel("Duration (seconds)") # set y label
    axs[1, 1].grid(True) # turn grid overlay on

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # adjust size of plot
    plt.show() # display the plots

# Analyze file size differences (2 subplots)
def visualize_file_sizes(sizes):
    # convert from bytes to kilobytes
    sizes_kb = [s / 1024 for s in sizes] # calc byte to kb
    df = pd.DataFrame({"size_kb": sizes_kb}) # save size in dataframe

    # create 1x2 subplot
    fig, axs = plt.subplots(1, 2, figsize=(14, 5)) # create plot w/ subplots
    fig.suptitle("Audio File Size Analysis", fontsize=16) # create subtitle

    # create a histogram
    sns.histplot(df["size_kb"], bins=30, kde=True, ax=axs[0]) # create the histogram
    axs[0].set_title("File Sizes") # set title
    axs[0].set_xlabel("File Size (KB)") # set x label
    axs[0].set_ylabel("Count") # set y label
    axs[0].grid(True) # turn grid overlay on

    # create a scatter plot
    axs[1].scatter(
        df.index, # x axis: file index
        df["size_kb"],# y axis: size in KB
        alpha=0.7 # adjust opacity
    )
    axs[1].set_title("Scatter of File Sizes") # set title
    axs[1].set_xlabel("File Index") # set x label
    axs[1].set_ylabel("File Size (KB)") # set y label
    axs[1].grid(True) # turn grid overlay on

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # adjust size of plot
    plt.show() # display the plots

# function to check fold distribution
def visualize_file_count(files_per_fold):
    # convert fold info to pandas dataframe
    df = pd.DataFrame(list(files_per_fold.items()), columns=["Fold", "File Count"]) # conversion

    # create a bar graph for the files of each fold
    plt.figure(figsize=(10, 5)) # create plot
    sns.barplot(x="Fold", y="File Count", data=df) # make it a bar graph
    plt.title("Number of Files per Fold") # set title
    plt.xlabel("Fold Number") # set x label
    plt.ylabel("File Count") # set y label
    plt.grid(True) # turn grid overlay on
    plt.tight_layout() # adjust layout
    plt.show() # display plot

# function to run all of the above visualizations
def visualize_audio_stats(durations, sizes, labels, label_names, files_per_fold):
    # show file/class distribution
    visualize_file_count(files_per_fold)

    # visualize audio file durations to ensure uniformity
    visualize_audio_durations(durations, labels, label_names)

    # visualize file size differences
    visualize_file_sizes(sizes)
