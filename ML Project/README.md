# Mall Customer Segmentation - Machine Learning Project

A comprehensive machine learning project that segments mall customers using K-Means clustering and Logistic Regression classification with regularization techniques.

## Project Overview

This project demonstrates a complete ML workflow combining:
- **Unsupervised Learning**: K-Means clustering to discover natural customer groupings
- **Supervised Learning**: Logistic Regression to classify customers into segments
- **Regularization**: L1 (Lasso) and L2 (Ridge) techniques to prevent overfitting
- **Optimization**: Elbow Method for optimal cluster selection

## Course Information
- **Course**: AIL 301 - Machine Learning
- **Semester**: Spring 2026
- **Institution**: Bahria University, Karachi Campus
- **Department**: Software Engineering

## Project Structure

```
ML Project/
├── archive/
│   └── Mall_Customers.csv          # Dataset
├── 01_data_preprocessing.py        # Module 1: Data exploration & preprocessing
├── 02_clustering.py                # Module 2: K-Means clustering
├── 03_classification.py            # Module 3: Logistic Regression & Regularization
├── 04_evaluation.py                # Module 4: Evaluation & reporting
├── main.py                         # Main execution script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset

The dataset is pre-loaded in the `archive/` folder. It contains:
- **Records**: ~200 customers
- **Features**: Age, Annual Income (k$), Spending Score (1-100)
- **Format**: CSV

## Running the Project

### Complete Pipeline (Recommended)

```bash
python -m streamlit run app.py
python main.py

This will execute all modules sequentially:
1. Load and explore the dataset
2. Perform K-Means clustering with Elbow Method
3. Train Logistic Regression models (L1, L2, and no regularization)
4. Evaluate and compare models
5. Generate visualizations and reports

### Individual Module Execution

You can also run individual modules:

```bash
# Data preprocessing only
python 01_data_preprocessing.py

# Clustering only
python 02_clustering.py

# Classification only
python 03_classification.py
```

## Output Files

### Visualizations
- `01_feature_exploration.png` - Feature distributions and relationships
- `02_elbow_method.png` - Optimal cluster determination curve
- `02_kmeans_clusters.png` - Customer segments in 2D space
- `03_regularization_comparison.png` - Model performance comparison
- `03_confusion_matrices.png` - Classification accuracy matrices
- `04_demographic_analysis.png` - Segment characteristics

### Reports & Data
- `04_project_report.txt` - Comprehensive project report with findings
- `04_segment_summary.csv` - Statistical summary of each segment
- `04_evaluation_summary.csv` - Model evaluation metrics

## Key Concepts

### Module 1: Data Preprocessing
- Load and explore customer data
- Analyze feature distributions
- Prepare data for ML algorithms
- Statistical summaries and visualizations

### Module 2: K-Means Clustering
- Apply K-Means algorithm to discover customer groups
- Use Elbow Method to find optimal k value
- Visualize clusters in 2D feature space
- Analyze cluster characteristics

### Module 3: Classification & Regularization
- Train Logistic Regression classifier
- Implement L1 (Lasso) regularization: encourages feature sparsity
- Implement L2 (Ridge) regularization: penalizes large coefficients
- Compare regularization techniques on test data
- Evaluate using accuracy, precision, recall, and F1-score

### Module 4: Evaluation & Reporting
- Generate comprehensive evaluation report
- Extract business insights from segments
- Compare model performances
- Create visualizations and data summaries

## Customer Segments

The K-Means clustering identifies 4 primary customer segments:

1. **High Value Customers** (High Income + High Spending)
   - Premium customers with high purchasing power and spending tendency
   - Target: Exclusive products, premium services

2. **Cautious High-Income** (High Income + Low Spending)
   - Wealthy but selective shoppers
   - Target: Quality-focused promotions, loyalty programs

3. **Enthusiastic Middle-Income** (Medium Income + High Spending)
   - Active shoppers with balanced finances
   - Target: Frequent promotions, variety offerings

4. **Budget-Conscious** (Low Income + Low Spending)
   - Price-sensitive customers
   - Target: Discounts, budget-friendly options

## Model Performance

The project compares three classification approaches:
- **L2 Regularization**: Smooth penalty, good generalization
- **L1 Regularization**: Sparse features, feature selection
- **No Regularization**: Baseline model

Evaluation metrics:
- Accuracy: Correct predictions
- Precision: True positives among predictions
- Recall: True positives detected
- F1-Score: Harmonic mean of precision and recall

## Technologies Used

- **Python 3.x**: Core language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: ML algorithms and metrics
- **Matplotlib**: Visualization and plotting
- **Seaborn**: Statistical data visualization

## How Regularization Works

**L1 Regularization (Lasso)**:
- Adds penalty: λ × Σ|coefficients|
- Encourages sparsity (some coefficients → 0)
- Performs automatic feature selection
- Useful when you suspect many features are irrelevant

**L2 Regularization (Ridge)**:
- Adds penalty: λ × Σ(coefficients²)
- Penalizes large coefficients evenly
- Reduces model complexity smoothly
- Handles multicollinearity better

**Trade-off Analysis**:
- Stronger regularization (lower C) → simpler model, less overfitting
- Weaker regularization (higher C) → more complex model, better training fit

## Business Insights

1. **Customer segmentation enables targeted marketing**
   - Different segments have different needs
   - Tailor promotions to segment characteristics

2. **Predictive classification for new customers**
   - Automatically assign new customers to segments
   - No need to re-run clustering

3. **Regularization prevents overfitting**
   - L1 and L2 improve model generalization
   - Better performance on unseen data

4. **Segment drift monitoring**
   - Periodically re-cluster to detect market changes
   - Adapt strategies to evolving customer base

## Future Enhancements

- Real-time customer segmentation
- Integration with CRM systems
- Automated segment-based email campaigns
- GUI dashboard for visualization
- A/B testing framework for promotions
- Deep learning approaches for complex patterns

## References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning" (2nd ed.). Springer.
2. Bishop, C. M. (2006). "Pattern Recognition and Machine Learning". Springer.
3. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, 12, 2825-2830.
4. Kaggle Dataset: [Mall Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## Authors

- Muhammad Umer (02-131222-031)
- Muhammad Furqan (02-131222-016)

**Submitted to**: Engr. Hina Shakir (Course Instructor), Engr. Muniba (Lab Instructor)

**Date**: May 11, 2026

## License

This project is created for educational purposes as part of the Machine Learning course curriculum.

---

**For questions or issues, please refer to the generated reports and visualizations for detailed analysis.**
