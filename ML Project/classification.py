"""
Module 3: Logistic Regression with L1/L2 Regularization
This module implements classification using Logistic Regression with different regularization techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve)

class LogisticRegressionClassifier:
    """Handles Logistic Regression classification with L1/L2 regularization."""

    def __init__(self, features, labels):
        """Initialize with features and cluster labels."""
        self.features = features
        self.labels = labels
        self.scaler = StandardScaler()
        self.features_scaled = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}  # Store different models (L1, L2, None)
        self.results = {}  # Store evaluation results

    def prepare_data(self, test_size=0.2):
        """
        Standardize features and split data.

        Args:
            test_size: Proportion of data for testing (default: 0.2)
        """
        print("=" * 60)
        print("DATA PREPARATION FOR CLASSIFICATION")
        print("=" * 60)

        # Standardize features
        print("Scaling features...")
        self.features_scaled = self.scaler.fit_transform(self.features)

        # Split data
        print(f"Splitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features_scaled, self.labels, test_size=test_size, random_state=42, stratify=self.labels
        )

        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}\n")

    def train_model(self, regularization='l2', C=1.0):
        """
        Train Logistic Regression model.

        Args:
            regularization: Type of regularization ('l1', 'l2', or None)
            C: Inverse regularization strength (lower C = stronger regularization)
        """
        print(f"Training Logistic Regression ({regularization} regularization, C={C})...")

        if regularization == 'l1':
            model = LogisticRegression(penalty='l1', solver='saga', C=C,
                                     random_state=42, max_iter=1000)
        elif regularization == 'l2':
            model = LogisticRegression(penalty='l2', solver='lbfgs', C=C,
                                     random_state=42, max_iter=1000)
        else:
            model = LogisticRegression(penalty=None, solver='lbfgs',
                                     random_state=42, max_iter=1000)

        model.fit(self.X_train, self.y_train)
        self.models[regularization] = model
        print(f"✓ Model trained successfully!\n")

        return model

    def evaluate_model(self, regularization='l2'):
        """
        Evaluate the trained model.

        Args:
            regularization: Type of regularization to evaluate
        """
        if regularization not in self.models:
            print(f"Model with {regularization} regularization not found!")
            return

        model = self.models[regularization]

        print("=" * 60)
        print(f"EVALUATION - {regularization.upper()} REGULARIZATION")
        print("=" * 60)

        # Predictions
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)

        # Calculate metrics
        metrics = {
            'Train Accuracy': accuracy_score(self.y_train, y_pred_train),
            'Test Accuracy': accuracy_score(self.y_test, y_pred_test),
            'Precision': precision_score(self.y_test, y_pred_test, average='weighted'),
            'Recall': recall_score(self.y_test, y_pred_test, average='weighted'),
            'F1-Score': f1_score(self.y_test, y_pred_test, average='weighted'),
        }

        self.results[regularization] = {
            'metrics': metrics,
            'y_pred': y_pred_test,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred_test),
            'classification_report': classification_report(self.y_test, y_pred_test)
        }

        # Print results
        print(f"\nTrain Accuracy:  {metrics['Train Accuracy']:.4f}")
        print(f"Test Accuracy:   {metrics['Test Accuracy']:.4f}")
        print(f"Precision:       {metrics['Precision']:.4f}")
        print(f"Recall:          {metrics['Recall']:.4f}")
        print(f"F1-Score:        {metrics['F1-Score']:.4f}")

        print(f"\nClassification Report:")
        print(self.results[regularization]['classification_report'])

        print(f"Confusion Matrix:")
        print(self.results[regularization]['confusion_matrix'])
        print()

    def compare_regularization(self):
        """Compare performance of different regularization techniques."""
        print("\n" + "=" * 60)
        print("REGULARIZATION COMPARISON")
        print("=" * 60)

        if not self.results:
            print("No models evaluated yet!")
            return

        print("\nSummary of Results:\n")
        comparison_df = pd.DataFrame({
            reg_type: metrics['metrics']
            for reg_type, results in self.results.items()
            for metrics in [results]
        }).T

        print(comparison_df.to_string())

        # Visualize comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics_to_plot = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score']
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            values = [self.results[reg]['metrics'][metric] for reg in self.results.keys()]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            ax.bar(self.results.keys(), values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.1])
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10)

        plt.suptitle('Regularization Technique Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('03_regularization_comparison.png', dpi=300, bbox_inches='tight')
        print("\nComparison plot saved as '03_regularization_comparison.png'")
        plt.close()

    def visualize_confusion_matrices(self):
        """Visualize confusion matrices for all models."""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))

        if n_models == 1:
            axes = [axes]

        for ax, (reg_type, results) in zip(axes, self.results.items()):
            cm = results['confusion_matrix']
            im = ax.imshow(cm, cmap='Blues', alpha=0.8)
            ax.set_title(f'Confusion Matrix ({reg_type.upper()})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')

            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text = ax.text(j, i, cm[i, j], ha="center", va="center",
                                 color="white" if cm[i, j] > cm.max()/2 else "black",
                                 fontsize=14, fontweight='bold')

            ax.set_xticks(range(cm.shape[0]))
            ax.set_yticks(range(cm.shape[0]))

        plt.tight_layout()
        plt.savefig('03_confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("Confusion matrices saved as '03_confusion_matrices.png'")
        plt.close()

    def get_model(self, regularization='l2'):
        """Return trained model."""
        return self.models.get(regularization)


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    from clustering import KMeansClustering

    preprocessor = DataPreprocessor("archive/Mall_Customers.csv")
    preprocessor.load_data()
    preprocessor.preprocess_data()
    features = preprocessor.get_features_for_clustering()

    clustering = KMeansClustering(features)
    clustering.scale_features()
    clustering.fit_kmeans(n_clusters=4)
    labels = clustering.get_labels()

    # Classification
    classifier = LogisticRegressionClassifier(features, labels)
    classifier.prepare_data()

    # Train with different regularization techniques
    for reg_type in ['l2', 'l1', None]:
        classifier.train_model(regularization=reg_type if reg_type else 'none')
        classifier.evaluate_model(regularization=reg_type if reg_type else 'none')

    classifier.compare_regularization()
    classifier.visualize_confusion_matrices()

    print("\n✓ Logistic Regression classification completed successfully!")
