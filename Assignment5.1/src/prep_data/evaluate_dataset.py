import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# function to plot class and file distribution (per fold)
def plot_dataset(csv_path):
    df = pd.read_csv(csv_path) # read csv

    # Get files per fold
    files_per_fold = df['fold'].value_counts().sort_index()

    # create dataframe
    fold_df = pd.DataFrame({
        "Fold": files_per_fold.index.astype(str),
        "File Count": files_per_fold.values
    })

    # Get class distribution
    class_counts = df['class'].value_counts().sort_index()

    # create subplots
    _, axes = plt.subplots(1, 2, figsize=(18, 6))

    # plot file distribution as bar graph
    sns.barplot(x="Fold", y="File Count", data=fold_df, ax=axes[0]) # create bar graph
    axes[0].set_title("Number of Files per Fold") # add title
    axes[0].set_xlabel("Fold Number") # add x label
    axes[0].set_ylabel("File Count") # add y label
    axes[0].grid(True) # add grid overlay

    # plot class distribution as bar 
    class_counts.plot(kind='bar', color='blue', edgecolor='black', ax=axes[1]) # plot class distr
    axes[1].set_title("Class Distribution") # add title
    axes[1].set_xlabel("Class") # add x label
    axes[1].set_ylabel("Number of Samples") # add y label
    axes[1].tick_params(axis='x', rotation=45) # adjust labels to 45 deg

    plt.tight_layout() # adjust layout
    plt.show() # display plot