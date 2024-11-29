from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


def visualize_target_distribution(data, target_column):
    y = data[target_column]

    # Plot Distribution (Histogram + KDE)
    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True, bins=50)
    plt.title(f"Distribution of {target_column}")
    plt.xlabel(target_column)
    plt.show()

    # Plot Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=y)
    plt.title(f"Box Plot of {target_column}")
    plt.xlabel(target_column)
    plt.show()


# Visualize PCA Explained Variance
def visualize_variance_PCA(pca, target_variance=None):
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    # Find the number of components needed for 90% variance
    if target_variance:
        n_components = np.argmax(cumulative_variance >= target_variance) + 1
        print(
            f"Number of components to retain {target_variance * 100}% variance: {n_components}"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance)
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid()
    plt.show()
