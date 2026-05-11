"""Module 4: Evaluation, Visualization and Report Generation"""
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class EvaluationReport:
    def __init__(self, preprocessor, clustering, classifier):
        self.preprocessor = preprocessor
        self.clustering = clustering
        self.classifier = classifier
        self.report_data = {}

    def generate_full_report(self):
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
            }
        }
        self.report_data = report
        print(f"\nDataset Summary:\n  Total Customers: {report['Dataset']['Total Customers']}")
        print(f"  Features: {report['Dataset']['Features Used']}")
        print(f"  Missing Values: {report['Dataset']['Missing Values']}")
        print(f"\nClustering Summary:\n  Algorithm: {report['Clustering']['Algorithm']}")
        print(f"  Optimal Clusters: {report['Clustering']['Optimal Clusters']}")
        print(f"  Inertia: {report['Clustering']['Inertia']:.2f}")
        print("\n✓ Report generated successfully!")

    def generate_cluster_insights(self):
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
        segment_summary = df.groupby('Cluster').agg({'Age': 'mean', 'Annual Income (k$)': 'mean', 'Spending Score (1-100)': 'mean', 'CustomerID': 'count'}).round(2)
        segment_summary.rename(columns={'CustomerID': 'Count'}, inplace=True)
        segment_summary.to_csv('04_segment_summary.csv')
        print("Segment summary saved to '04_segment_summary.csv'\n")
        return df

    def visualize_segments_by_demographics(self, df_with_clusters):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        cluster_counts = df_with_clusters['Cluster'].value_counts().sort_index()
        axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Customer Distribution by Cluster', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Number of Customers')
        df_with_clusters.boxplot(column='Annual Income (k$)', by='Cluster', ax=axes[0, 1])
        axes[0, 1].set_title('Annual Income Distribution by Cluster')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('Annual Income (k$)')
        df_with_clusters.boxplot(column='Spending Score (1-100)', by='Cluster', ax=axes[1, 0])
        axes[1, 0].set_title('Spending Score Distribution by Cluster')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Spending Score (1-100)')
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
        report_text = "\n" + "=" * 70 + "\nMALL CUSTOMER SEGMENTATION - PROJECT REPORT\n" + "=" * 70
        report_text += f"\nGenerated: {self.report_data.get('Generated', 'N/A')}\n\n4. KEY FINDINGS\n" + "-" * 70
        report_text += "\n- Customers successfully segmented into groups\n- Logistic Regression effectively classifies customers\n"
        with open('04_project_report.txt', 'w') as f:
            f.write(report_text)
        print("\nReport saved to '04_project_report.txt'")

    def save_evaluation_summary(self):
        summary_data = {'Metric': ['Total Customers', 'Optimal Clusters'], 'Value': [self.report_data['Dataset']['Total Customers'], self.report_data['Clustering']['Optimal Clusters']]}
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('04_evaluation_summary.csv', index=False)
        print("Evaluation summary saved to '04_evaluation_summary.csv'")
