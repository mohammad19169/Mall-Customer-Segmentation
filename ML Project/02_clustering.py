"""
Module 2: K-Means Clustering & Elbow Method
This module implements K-Means clustering and determines optimal cluster count.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class KMeansClustering:
    """Handles K-Means clustering and Elbow Method analysis."""

    def __init__(self, features):
        """Initialize with feature data."""
        self.features = features
        self.scaler = StandardScaler()
        self.features_scaled = None
        self.kmeans = None
        self.labels = None
        self.inertias = []
        self.optimal_k = None

    def scale_features(self):
        """Standardize features for better clustering."""
        print("Scaling features...")
        self.features_scaled = self.scaler.fit_transform(self.features)
        print(f"Features scaled. Shape: {self.features_scaled.shape}\n")
        return self.features_scaled

    def elbow_method(self, k_range=range(1, 11)):
        """
        Apply Elbow Method to find optimal number of clusters.

        Args:
            k_range: Range of k values to test (default: 1-10)
        """
        print("=" * 60)
        print("ELBOW METHOD - FINDING OPTIMAL CLUSTERS")
        print("=" * 60)

        self.inertias = []
        k_values = list(k_range)

        print(f"Testing cluster counts: {k_values}")
        for k in k_values:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(self.features_scaled)
            self.inertias.append(kmeans_temp.inertia_)
            print(f"k={k}: Inertia = {kmeans_temp.inertia_:.2f}")

        # Visualize Elbow Method
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, self.inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_values)
        plt.savefig('02_elbow_method.png', dpi=300, bbox_inches='tight')
        print("\nElbow Method plot saved as '02_elbow_method.png'")
        plt.close()

        return k_values, self.inertias

    def fit_kmeans(self, n_clusters):
        """
        Fit K-Means with specified number of clusters.

        Args:
            n_clusters: Number of clusters
        """
        print(f"\n" + "=" * 60)
        print(f"FITTING K-MEANS WITH {n_clusters} CLUSTERS")
        print("=" * 60)

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = self.kmeans.fit_predict(self.features_scaled)

        print(f"Cluster centers (scaled): \n{self.kmeans.cluster_centers_}\n")
        print(f"Cluster distribution:")
        unique, counts = np.unique(self.labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} customers ({count/len(self.labels)*100:.1f}%)")

        self.optimal_k = n_clusters
        return self.labels

    def visualize_clusters(self, features_original=None):
        """
        Visualize clusters in 2D space.

        Args:
            features_original: Original unscaled features for better interpretation
        """
        print(f"\nVisualizing {self.optimal_k} clusters...")

        plt.figure(figsize=(12, 6))

        # Plot clusters
        scatter = plt.scatter(self.features_scaled[:, 0], self.features_scaled[:, 1],
                            c=self.labels, cmap='viridis', alpha=0.6, s=100, edgecolors='black')

        # Plot cluster centers
        plt.scatter(self.kmeans.cluster_centers_[:, 0],
                   self.kmeans.cluster_centers_[:, 1],
                   c='red', marker='X', s=300, edgecolors='black', linewidth=2,
                   label='Centroids')

        plt.xlabel('Scaled Annual Income (k$)', fontsize=12)
        plt.ylabel('Scaled Spending Score (1-100)', fontsize=12)
        plt.title(f'Customer Segments (K-Means with k={self.optimal_k})', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('02_kmeans_clusters.png', dpi=300, bbox_inches='tight')
        print("Cluster visualization saved as '02_kmeans_clusters.png'")
        plt.close()

    def get_labels(self):
        """Return cluster labels."""
        return self.labels

    def get_cluster_centers(self):
        """Return cluster centers (scaled)."""
        return self.kmeans.cluster_centers_

    def get_inertia(self):
        """Return inertia of fitted model."""
        return self.kmeans.inertia_ if self.kmeans else None


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor("archive/Mall_Customers.csv")
    preprocessor.load_data()
    preprocessor.preprocess_data()
    features = preprocessor.get_features_for_clustering()

    clustering = KMeansClustering(features)
    clustering.scale_features()
    clustering.elbow_method()
    clustering.fit_kmeans(n_clusters=4)
    clustering.visualize_clusters()

    print("\n✓ K-Means clustering completed successfully!")
