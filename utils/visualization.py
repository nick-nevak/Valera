from matplotlib import pyplot as plt
import seaborn as sns


def visualize_target_distribution(data, target_column):
    y = data[target_column]

    # Plot Distribution (Histogram + KDE)
    plt.figure(figsize=(10, 6))
    sns.histplot(y, kde=True, bins=50)
    plt.title(f'Distribution of {target_column}')
    plt.xlabel(target_column)
    plt.show()

    # Plot Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=y)
    plt.title(f'Box Plot of {target_column}')
    plt.xlabel(target_column)
    plt.show()
