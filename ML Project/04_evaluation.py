"""
Module 4: Evaluation, Visualization & Report Generation
This module generates comprehensive evaluation and visualizations for the entire project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class EvaluationReport:
    """Generates comprehensive evaluation reports and visualizations."""

    def __init__(self, preprocessor, clustering, classifier):
        """Initialize with all components."""
        self.preprocessor = preprocessor
        self.clustering = clustering
        self.classifier = classifier
        self.report_data = {}

    def generate_full_report(self):
        """Generate comprehensive evaluation report."""
        print("=" * 60)
        print("GENERATING COMPREHENSIVE EVALUATION REPORT")
        print("=" * 60)

        report = {
            'Generated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Dataset': {
                'Total Customers': len(self.preprocessor.processed_df),
                'Features Used': 'Annual Income, Spending Score',
                'Missing Values': int(self.preprocessor.processed_df.isnull().sum().sum())
            },
            'Clustering': {
                'Algorithm': 'K-Means',
                'Optimal Clusters': self.clustering.optimal_k,
                'Inertia': self.clustering.get_inertia()
            },
            'Classification': self.classifier.results
        }

        self.report_data = report

        print(f"\nDataset Summary:")
        print(f"  Total Customers: {report['Dataset']['Total Customers']}")
        print(f"  Features: {report['Dataset']['Features Used']}")
        print(f"  Missing Values: {report['Dataset']['Missing Values']}")

        print(f"\nClustering Summary:")
        print(f"  Algorithm: {report['Clustering']['Algorithm']}")
        print(f"  Optimal Clusters: {report['Clustering']['Optimal Clusters']}")
        print(f"  Inertia: {report['Clustering']['Inertia']:.2f}")

        print("\n✓ Report generated successfully!")

    def generate_cluster_insights(self):
        """Generate business insights from clusters."""
        print("\n" + "=" * 60)
        print("CUSTOMER SEGMENT INSIGHTS")
        print("=" * 60)

        df = self.preprocessor.processed_df.copy()
        df['Cluster'] = self.clustering.get_labels()

        print("\nSegment Statistics:\n")

        for cluster_id in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster_id]
            print(f"Cluster {cluster_id} ({len(cluster_data)} customers):")
            print(f"  Avg Age: {cluster_data['Age'].mean():.1f} years")
            print(f"  Avg Annual Income: ${cluster_data['Annual Income (k$)'].mean():.1f}k")
            print(f"  Avg Spending Score: {cluster_data['Spending Score (1-100)'].mean():.1f}")

            # Segment characterization
            income = cluster_data['Annual Income (k$)'].mean()
            spending = cluster_data['Spending Score (1-100)'].mean()

            if income > 75 and spending > 50:
                segment_name = "HIGH VALUE - Premium Customers"
            elif income > 75 and spending <= 50:
                segment_name = "HIGH INCOME - Cautious Spenders"
            elif income <= 75 and spending > 50:
                segment_name = "MIDDLE INCOME - Enthusiastic Shoppers"
            else:
                segment_name = "LOW ENGAGEMENT - Budget Conscious"

            print(f"  Segment Type: {segment_name}\n")

        # Save segment data
        segment_summary = df.groupby('Cluster').agg({
            'Age': 'mean',
            'Annual Income (k$)': 'mean',
            'Spending Score (1-100)': 'mean',
            'CustomerID': 'count'
        }).round(2)
        segment_summary.rename(columns={'CustomerID': 'Count'}, inplace=True)

        segment_summary.to_csv('04_segment_summary.csv')
        print("Segment summary saved to '04_segment_summary.csv'\n")

        return df

    def visualize_segments_by_demographics(self, df_with_clusters):
        """Visualize customer segments with demographic analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Cluster distribution
        cluster_counts = df_with_clusters['Cluster'].value_counts().sort_index()
        axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Customer Distribution by Cluster', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Number of Customers')

        # Income by cluster
        df_with_clusters.boxplot(column='Annual Income (k$)', by='Cluster', ax=axes[0, 1])
        axes[0, 1].set_title('Annual Income Distribution by Cluster')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Annual Income (k$)')

        # Spending Score by cluster
        df_with_clusters.boxplot(column='Spending Score (1-100)', by='Cluster', ax=axes[1, 0])
        axes[1, 0].set_title('Spending Score Distribution by Cluster')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Spending Score (1-100)')

        # Age by cluster
        df_with_clusters.boxplot(column='Age', by='Cluster', ax=axes[1, 1])
        axes[1, 1].set_title('Age Distribution by Cluster')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Age')

        plt.suptitle('Demographic Analysis by Customer Segment', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('04_demographic_analysis.png', dpi=300, bbox_inches='tight')
        print("Demographic analysis saved as '04_demographic_analysis.png'")
        plt.close()

    def create_summary_report(self):
        """Create a text summary report."""
        report_text = f"""
{'=' * 70}
MALL CUSTOMER SEGMENTATION - COMPREHENSIVE PROJECT REPORT
{'=' * 70}

Generated: {self.report_data.get('Generated', 'N/A')}

1. DATASET OVERVIEW
{'-' * 70}
Total Customers: {self.report_data['Dataset']['Total Customers']}
Features Analyzed: {self.report_data['Dataset']['Features Used']}
Missing Values: {self.report_data['Dataset']['Missing Values']}

2. CLUSTERING ANALYSIS
{'-' * 70}
Algorithm Used: K-Means Clustering
Optimal Number of Clusters: {self.report_data['Clustering']['Optimal Clusters']}
Model Inertia (Within-cluster sum of squares): {self.report_data['Clustering']['Inertia']:.2f}

The Elbow Method was used to determine the optimal number of clusters.
This identifies the point where the rate of inertia decrease significantly
diminishes, balancing model complexity and explanatory power.

3. CLASSIFICATION PERFORMANCE
{'-' * 70}
Multiple Logistic Regression models were trained with different regularization
techniques to classify customers into pre-identified segments.

Models Compared:
"""

        if self.classifier.results:
            for reg_type, results in self.classifier.results.items():
                metrics = results['metrics']
                report_text += f"""
{reg_type.upper()} Regularization:
  - Train Accuracy: {metrics.get('Train Accuracy', 'N/A'):.4f}
  - Test Accuracy:  {metrics.get('Test Accuracy', 'N/A'):.4f}
  - Precision:      {metrics.get('Precision', 'N/A'):.4f}
  - Recall:         {metrics.get('Recall', 'N/A'):.4f}
  - F1-Score:       {metrics.get('F1-Score', 'N/A'):.4f}
"""

        report_text += f"""
4. KEY FINDINGS
{'-' * 70}
- Customers successfully segmented into {self.report_data['Clustering']['Optimal Clusters']} distinct groups
- Logistic Regression effectively classifies new customers into segments
- Regularization techniques help prevent overfitting and improve generalization
- Different regularization strengths offer trade-offs between bias and variance

5. RECOMMENDATIONS
{'-' * 70}
1. Use identified customer segments for targeted marketing campaigns
2. Apply the trained classification model to categorize new customers
3. Monitor segment drift over time and retrain clustering models periodically
4. Consider L2 regularization for balanced model complexity and performance
5. Implement segment-specific promotional strategies based on insights

6. GENERATED VISUALIZATIONS
{'-' * 70}
- 01_feature_exploration.png: Feature distributions and relationships
- 02_elbow_method.png: Optimal cluster determination
- 02_kmeans_clusters.png: Customer segments visualization
- 03_regularization_comparison.png: Model performance comparison
- 03_confusion_matrices.png: Classification accuracy details
- 04_demographic_analysis.png: Segment demographic characteristics
- 04_segment_summary.csv: Statistical summary of each segment

{'=' * 70}
"""

        # Save report
        with open('04_project_report.txt', 'w') as f:
            f.write(report_text)

        print("\n" + report_text)
        print("Full report saved to '04_project_report.txt'")

    def save_evaluation_summary(self):
        """Save evaluation summary to CSV."""
        summary_data = {
            'Metric': [],
            'Value': []
        }

        summary_data['Metric'].append('Total Customers')
        summary_data['Value'].append(self.report_data['Dataset']['Total Customers'])

        summary_data['Metric'].append('Optimal Clusters')
        summary_data['Value'].append(self.report_data['Clustering']['Optimal Clusters'])

        summary_data['Metric'].append('Model Inertia')
        summary_data['Value'].append(f"{self.report_data['Clustering']['Inertia']:.2f}")

        for reg_type, results in self.classifier.results.items():
            metrics = results['metrics']
            for metric_name, value in metrics.items():
                summary_data['Metric'].append(f"{reg_type.upper()} - {metric_name}")
                summary_data['Value'].append(f"{value:.4f}")

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('04_evaluation_summary.csv', index=False)
        print("\nEvaluation summary saved to '04_evaluation_summary.csv'")


if __name__ == "__main__":
    # Example usage - This would be called from main.py
    pass
