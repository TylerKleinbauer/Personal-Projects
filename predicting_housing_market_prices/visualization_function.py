import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df, figsize=(11,9), vmin=-1, vmax=1, center=0, fmt='.2f'):
    """
    Creates and displays a correlation heatmap for the given dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to calculate correlations for
    figsize : tuple, optional
        Figure size as (width, height), default (11,9)
    vmin : float, optional
        Minimum value for the colormap, default 0
    vmax : float, optional 
        Maximum value for the colormap, default 1
    center : float, optional
        Center value for the colormap, default 0
    fmt : str, optional
        Format string for annotation values, default '.2f'
    """
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(240,10, sep=20, as_cmap=True)

    sns.heatmap(corr,
                mask=mask,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center, 
                square=True,
                linewidths=.5,
                cbar_kws={'shrink':.5},
                annot=True,
                fmt=fmt)
    
    plt.show()


def plot_all_columns(df):
    """
    plots all numeric columns of a data frame with 'date' on the x-axis 
    """
    
    # Get numeric columns excluding the date column
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Create a figure with subplots arranged in a reasonable layout
    n_cols = 2  # Display 2 columns of plots
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Calculate needed rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()  # Flatten to make indexing easier

    # Plot each demographic metric
    for idx, column in enumerate(numeric_columns):
        axes[idx].plot(df['date'], df[column])
        axes[idx].set_title(column)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True)

    # Remove any empty subplots
    for idx in range(len(numeric_columns), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()