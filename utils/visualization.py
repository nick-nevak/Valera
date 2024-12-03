from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import NearestNeighbors


def visualize_distribution(x, name):
    # Plot Distribution (Histogram + KDE)
    plt.figure(figsize=(10, 6))
    sns.histplot(x, kde=True, bins=50)
    plt.title(f"Distribution of {name}")
    plt.xlabel(name)
    plt.show()


def visualize_box_plot(x, name):
    # Plot Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x)
    plt.title(f"Box Plot of {name}")
    plt.xlabel(name)
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


def visualize_k_neighbors_distance(X, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)

    # Plot the k-Distance
    k_distances = np.sort(distances[:, k - 1])

    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.title("k-Distance Plot")
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{k}-th Nearest Neighbor Distance")

    # Set granular y-axis ticks with a step of 5
    y_min, y_max = plt.ylim()  # Get current y-axis limits
    plt.yticks(np.arange(0, y_max + 1, 5))  # Set ticks from 0 to max with a step of 5

    plt.grid()
    plt.show()
