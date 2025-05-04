import pandas as pd
import matplotlib.pyplot as plt

# TODO: cleanup and comment
def plot_dataset(csv_path):
    df = pd.read_csv(csv_path)

    # Fold distribution
    files_per_fold = df['fold'].value_counts().sort_index()
    class_counts = df['class'].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # colors = plt.cm.tab10.colors # get diff colors for every fold or class

    # Fold Count bar graph
    axes[0].bar(
        files_per_fold.index.astype(str),
        files_per_fold.values,
        width=0.9,
        color='salmon',
        edgecolor='black'
    )
    axes[0].set_title("Number of Files per Fold")
    axes[0].set_xlabel("Fold Number")
    axes[0].set_ylabel("File Count")
    axes[0].grid(True)

    # Class distribution bar chart
    axes[1].bar(
        class_counts.index,
        class_counts.values,
        width=0.9,
        color='steelblue',
        edgecolor='black'
    )
    axes[1].set_title("Class Distribution")
    axes[1].set_ylabel("Number of Samples")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.show()