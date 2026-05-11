"""
Module 1: Data Preprocessing & Exploration
This module handles loading, exploring, and preparing the customer data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DataPreprocessor:
    """Handles data loading, exploration, and preprocessing."""

    def __init__(self, data_path):
        """Initialize with data path."""
        self.data_path = data_path
        self.df = None
        self.processed_df = None

    def load_data(self):
        """Load the dataset."""
        print(f"Loading dataset from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.df.shape}\n")
        return self.df

    def explore_data(self):
        """Explore and display basic statistics."""
        print("=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)

        print("\nFirst few rows:")
        print(self.df.head())

        print("\nDataset Info:")
        print(self.df.info())

        print("\nBasic Statistics:")
        print(self.df.describe())

        print("\nMissing Values:")
        print(self.df.isnull().sum())

        print("\nData Types:")
        print(self.df.dtypes)

    def preprocess_data(self):
        """Clean and prepare data for ML models."""
        print("\n" + "=" * 60)
        print("DATA PREPROCESSING")
        print("=" * 60)

        # Create a copy for processing
        self.processed_df = self.df.copy()

        # Select features for clustering (Annual Income and Spending Score)
        self.features_clustering = ['Annual Income (k$)', 'Spending Score (1-100)']

        # Check if columns exist
        print(f"\nAvailable columns: {self.df.columns.tolist()}")

        # Handle missing values if any
        if self.processed_df.isnull().sum().sum() > 0:
            print("Handling missing values...")
            self.processed_df = self.processed_df.dropna()

        print(f"Processed dataset shape: {self.processed_df.shape}")

        return self.processed_df

    def visualize_features(self):
        """Create visualizations of key features."""
        print("\n" + "=" * 60)
        print("FEATURE VISUALIZATION")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Annual Income distribution
        axes[0, 0].hist(self.processed_df['Annual Income (k$)'], bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Annual Income Distribution')
        axes[0, 0].set_xlabel('Annual Income (k$)')
        axes[0, 0].set_ylabel('Frequency')

        # Spending Score distribution
        axes[0, 1].hist(self.processed_df['Spending Score (1-100)'], bins=20, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Spending Score Distribution')
        axes[0, 1].set_xlabel('Spending Score (1-100)')
        axes[0, 1].set_ylabel('Frequency')

        # Age distribution
        axes[1, 0].hist(self.processed_df['Age'], bins=20, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Age Distribution')
        axes[1, 0].set_xlabel('Age')
        axes[1, 0].set_ylabel('Frequency')

        # Annual Income vs Spending Score scatter
        axes[1, 1].scatter(self.processed_df['Annual Income (k$)'],
                           self.processed_df['Spending Score (1-100)'],
                           alpha=0.6, color='purple')
        axes[1, 1].set_title('Annual Income vs Spending Score')
        axes[1, 1].set_xlabel('Annual Income (k$)')
        axes[1, 1].set_ylabel('Spending Score (1-100)')

        plt.tight_layout()
        plt.savefig('01_feature_exploration.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as '01_feature_exploration.png'")
        plt.close()

    def get_features_for_clustering(self):
        """Return features prepared for clustering."""
        return self.processed_df[self.features_clustering].values

    def get_processed_dataframe(self):
        """Return the processed dataframe."""
        return self.processed_df


if __name__ == "__main__":
    # Example usage
    data_path = "archive/Mall_Customers.csv"

    preprocessor = DataPreprocessor(data_path)
    preprocessor.load_data()
    preprocessor.explore_data()
    preprocessor.preprocess_data()
    preprocessor.visualize_features()

    print("\n✓ Data preprocessing completed successfully!")
