import pandas as pd
import os
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
visualisations_path = os.path.join(project_root, 'Clustering_Analysis', 'visualisations')


def optimise_k_means(data, max_k):
    """
    The function uses the elbow method to determine the optimal number of clusters for KMeans clustering.
    
    Args:
        data : The dataset to cluster.
        max_k : The maximum number of clusters to evaluate.
        visualisations_path : The path where the elbow plot will be saved.
    
    Returns:
        inertias (list): The inertia values for each K (within-cluster sum of squares).
    """
    # Ensure max_k is greater than 1
    if max_k < 2:
        raise ValueError("max_k should be at least 2 for clustering analysis.")
    
    means = []
    inertias = []
    
    # calculate KMeans inertia (WCSS)
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(data)
        
        means.append(k)
        inertias.append(kmeans.inertia_)
    
    # Generate the elbow plot
    plt.figure(figsize=(10, 5))
    plt.plot(means, inertias, 'o-', markersize=8, label='WCSS (Inertia)')
    plt.title('Elbow Method for Optimal K', fontsize=14)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('WCSS (Inertia)', fontsize=12)
    plt.grid(True)
    plt.legend()
    
    # Save and display the plot
    elbow_plot_path = os.path.join(visualisations_path, 'elbow_plot.png')
    plt.savefig(elbow_plot_path)
    plt.show()
    
    return inertias

    
# apply K means Clustering
        
def apply_k_means(scaled_df_n_clusters, data, n_clusters, visualisations_path):
    """
    Apply K_Means clustering on the dataset. The optimal number of cluster is
    determined by the elbow method.

    Args:
        scaled_df_n_clusters: The Min_max Scaled dataframe containing customer data.
        data: Contains the features used for clustering (tenure and MonthlyCharges).
        n_clusters: The optimal number of cluster determined from elbow method.
        visualisations_path : The path to save the visualisations 
    
    Returns:
        kmeans (KMeans): The trained KMeans model.
        scaled_df_n_clusters : The input dataframe with an additional column for cluster labels.
        
    """
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(data)
    
    # Add the cluster labels to the DataFrame
    scaled_df_n_clusters['clusters_4'] = kmeans.labels_
    print(scaled_df_n_clusters)
    
    # Scatter plot for clusters
    plt.scatter(x=scaled_df_n_clusters['tenure'], y=scaled_df_n_clusters['MonthlyCharges'], 
                c=scaled_df_n_clusters['clusters_4'], cmap='viridis')
    
    # Scatter plot for centroids
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                color='purple', marker='*', s=200, label='Centroid')
    
    # Label each centroid with its cluster number
    for i, centroid in enumerate(kmeans.cluster_centers_):
        plt.text(centroid[0], centroid[1], f'Cluster {i}', fontsize=12, color='black', 
                 ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))
    
    # Axis labels
    plt.xlabel('Tenure')
    plt.ylabel('Monthly Charges')
    plt.legend(loc='best')
    
    # Save the plot
    optimal_cluster_path = os.path.join(visualisations_path, 'optimal_clusters.png')
    plt.savefig(optimal_cluster_path)
    
    # Display the plot
    plt.show()
    return kmeans, scaled_df_n_clusters
    
def save_and_display_plot(plt, file_path, title, xlabel, ylabel):
    """
    Function to save and display plots.
    
    Args:
        plt       : The current plot.
        file_path : The file path to save the plot.
        title     : The title of the plot.
        xlabel    : The label for the x-axis.
        ylabel    : The label for the y-axis.
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()

# Visulaise the cluster chatacterstics

def summarize_and_visualize_clusters(scaled_df_n_clusters, kmeans, visualisations_path):
    """
    Visualise cluster characteristics and save the plot to visualisation path.

    Args:
        scaled_df_n_clusters : The dataframe containing customer data and cluster assignments.
        kmeans (KMeans): The trained KMeans model.
        visualisations_path : The path to save visualizations.
    """
    
    # Assign cluster labels to the DataFrame
    scaled_df_n_clusters['clusters_4'] = kmeans.labels_

    # Summarize key metrics (mean values for MonthlyCharges and Tenure) by cluster
    cluster_summary = scaled_df_n_clusters.groupby('clusters_4').agg({
        'MonthlyCharges': ['mean'],
        'tenure': ['mean'],
    })

    # --- Visualization 1: Average Monthly Charges by Cluster ---
    cluster_summary['MonthlyCharges'].plot(kind='bar', legend=False)
    avg_monthly_charge_path = os.path.join(visualisations_path, 'average_monthly_charge.png')
    save_and_display_plot(plt, avg_monthly_charge_path, 
                          title='Average Monthly Charges by Cluster', 
                          xlabel='Cluster', ylabel='Average Monthly Charges')

    # --- Visualization 2: Average Tenure by Cluster ---
    cluster_summary['tenure'].plot(kind='bar', legend=False)
    avg_tenure_path = os.path.join(visualisations_path, 'average_tenure.png')
    save_and_display_plot(plt, avg_tenure_path, 
                          title='Average Tenure by Cluster', 
                          xlabel='Cluster', ylabel='Average Tenure')

    # --- Visualization 3: Cluster Distribution ---
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette('viridis', 4)
    # Map the colors to clusters
    cluster_colors = {i: color for i, color in enumerate(palette)}
    sns.countplot(x='clusters_4', hue='clusters_4', data=scaled_df_n_clusters, 
              palette=[cluster_colors[i] for i in sorted(cluster_colors.keys())], legend=False)
    cluster_distribution_path = os.path.join(visualisations_path, 'cluster_distribution.png')
    save_and_display_plot(plt, cluster_distribution_path, 
                          title='Cluster Distribution', 
                          xlabel='Cluster', ylabel='Count')


