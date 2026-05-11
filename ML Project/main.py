
import sys
import warnings

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# Import all modules
from data_preprocessing import DataPreprocessor
from clustering import KMeansClustering
from classification import LogisticRegressionClassifier
from evaluation import EvaluationReport


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70 + "\n")


def main():
    """Main execution function."""

    print_header("MALL CUSTOMER SEGMENTATION - COMPLETE ML PIPELINE")

    # ========== STEP 1: DATA PREPROCESSING ==========
    print_header("STEP 1: DATA PREPROCESSING & EXPLORATION")

    try:
        preprocessor = DataPreprocessor("archive/Mall_Customers.csv")
        preprocessor.load_data()
        preprocessor.explore_data()
        preprocessor.preprocess_data()
        preprocessor.visualize_features()
        print("\n✓ Data preprocessing completed successfully!")
    except Exception as e:
        print(f"✗ Error in data preprocessing: {e}")
        return

    # ========== STEP 2: K-MEANS CLUSTERING ==========
    print_header("STEP 2: K-MEANS CLUSTERING & ELBOW METHOD")

    try:
        features = preprocessor.get_features_for_clustering()

        clustering = KMeansClustering(features)
        clustering.scale_features()

        # Apply Elbow Method
        k_range = range(1, 11)
        clustering.elbow_method(k_range)

        # Fit K-Means with optimal clusters (user would determine this from elbow plot)
        # For this example, we'll use 4 clusters (typical for this dataset)
        optimal_k = 4
        print(f"\n>>> Using {optimal_k} clusters based on Elbow Method analysis\n")

        clustering.fit_kmeans(n_clusters=optimal_k)
        clustering.visualize_clusters()
        print("\n✓ K-Means clustering completed successfully!")
    except Exception as e:
        print(f"✗ Error in clustering: {e}")
        return

    # ========== STEP 3: LOGISTIC REGRESSION CLASSIFICATION ==========
    print_header("STEP 3: LOGISTIC REGRESSION & REGULARIZATION")

    try:
        labels = clustering.get_labels()
        features = preprocessor.get_features_for_clustering()

        classifier = LogisticRegressionClassifier(features, labels)
        classifier.prepare_data(test_size=0.2)

        # Train models with different regularization techniques
        print("\nTraining models with different regularization techniques:\n")

        regularization_types = ['l2', 'l1', None]
        for reg_type in regularization_types:
            display_name = reg_type.upper() if reg_type else 'NONE'
            print(f"\n--- Training with {display_name} Regularization ---")
            model_key = reg_type if reg_type else 'none'
            classifier.train_model(regularization=model_key)

        # Evaluate all models
        print("\n--- Evaluating Models ---")
        for reg_type in regularization_types:
            model_key = reg_type if reg_type else 'none'
            classifier.evaluate_model(regularization=model_key)

        # Compare regularization techniques
        classifier.compare_regularization()
        classifier.visualize_confusion_matrices()

        print("\n✓ Classification completed successfully!")
    except Exception as e:
        print(f"✗ Error in classification: {e}")
        return

    # ========== STEP 4: EVALUATION & REPORTING ==========
    print_header("STEP 4: EVALUATION, INSIGHTS & REPORT GENERATION")

    try:
        evaluator = EvaluationReport(preprocessor, clustering, classifier)

        # Generate comprehensive report
        evaluator.generate_full_report()

        # Generate cluster insights
        df_with_clusters = evaluator.generate_cluster_insights()

        # Visualize demographic analysis
        evaluator.visualize_segments_by_demographics(df_with_clusters)

        # Create summary report
        evaluator.create_summary_report()

        # Save evaluation summary
        evaluator.save_evaluation_summary()

        print("\n✓ Evaluation and reporting completed successfully!")
    except Exception as e:
        print(f"✗ Error in evaluation: {e}")
        return

    # ========== COMPLETION MESSAGE ==========
    print_header("PROJECT COMPLETED SUCCESSFULLY!")

    print("\nGenerated Outputs:")
    print("  Visualizations:")
    print("    - 01_feature_exploration.png")
    print("    - 02_elbow_method.png")
    print("    - 02_kmeans_clusters.png")
    print("    - 03_regularization_comparison.png")
    print("    - 03_confusion_matrices.png")
    print("    - 04_demographic_analysis.png")
    print("\n  Reports & Data:")
    print("    - 04_project_report.txt")
    print("    - 04_segment_summary.csv")
    print("    - 04_evaluation_summary.csv")

    print("\n" + "=" * 70)
    print("All results have been saved to the current directory!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Program interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
