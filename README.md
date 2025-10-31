# Insurance Claim Severity and Frequency Prediction using Neural Networks

## Project Overview

This project develops predictive models for insurance claim severity analysis using both traditional machine learning approaches and advanced Neural Network architectures. The primary objective is to accurately predict **claim severity** - the cost/amount of individual insurance claims given that a claim has occurred.

**Key Focus**: Severity-only modeling using the Allstate Insurance dataset, which contains exclusively filed claims (no non-claimant records). This is critical for:
- Accurate reserve estimation
- Portfolio risk assessment  
- Large claim identification and prioritization
- Pricing optimization for renewal business

**Secondary Objective**: Binary classification of claims into high-cost vs. low-cost categories for claim triage and resource allocation.

## Business Context

Accurate claim prediction enables insurance companies to:
- Price policies appropriately based on risk
- Allocate reserves effectively
- Identify high-risk segments
- Optimize underwriting decisions
- Improve profitability while remaining competitive

## Dataset

**Source**: Allstate Insurance Claims Dataset  
**Size**: 188,318 claim records  
**Features**: 132 total features
- 116 categorical features (87.88%) - including policyholder characteristics, coverage details, and risk factors
- 14 continuous features (10.61%) - numerical risk indicators and policy attributes
- 1 ID column
- 1 target variable (loss amount in dollars)

**Target Variable**: Claim Severity (Loss Amount)
- **Mean**: $3,037
- **Median**: $2,116
- **Range**: $0.67 - $121,012
- **Distribution**: Highly right-skewed (skewness = 3.79)
- **Note**: Dataset contains only claims that occurred (no zero losses), making this a pure severity prediction problem

## Methodology

### 1. Baseline Model: XGBoost

Traditional gradient boosting approach for establishing performance benchmarks:

**Why XGBoost:**
- Handles high-cardinality categorical features efficiently
- Robust to outliers and skewed distributions
- Provides feature importance insights
- Industry-proven performance on tabular data

**Implementation:**
- Log-transformation of target variable (to address extreme skewness)
- Native categorical feature handling
- Hyperparameter tuning via cross-validation

### 2. Neural Feature Synthesis (NeurFS)

Advanced neural network approach specifically designed for high-dimensional categorical data:

**Architecture:**
- **Embedding layers** for high-cardinality categorical features (especially cat116 with 326 unique values)
- Deep neural networks with multiple hidden layers
- Batch normalization and dropout for regularization
- ReLU activation functions

**Key Innovation - Handling High Cardinality:**
Rather than one-hot encoding (which would create 326+ sparse columns), NeurFS uses learned embeddings that:
- Compress categorical features into dense representations
- Capture non-linear relationships automatically
- Prevent memory explosion with high-cardinality features

**Advantages:**
- Discovers complex feature interactions
- Handles extreme non-linearity (weak linear correlations in data)
- Scales efficiently with large datasets (188K records)
- Learns optimal feature representations

## Key Features

### Data Quality & Exploration
- **Clean dataset**: 0 missing values, 0 duplicates across 188,318 records
- **Comprehensive EDA**: Distribution analysis, correlation studies, feature-target relationships
- **Challenge identification**: Extreme skewness (3.79), high cardinality (cat116 = 326 values), weak linear correlations (max 0.14)

### Advanced Preprocessing
- **Log transformation** of target variable to address extreme right skew
- **Neural embeddings** for high-cardinality categorical features (avoiding memory explosion from one-hot encoding)
- **Stratified splitting**: 60/20/20 train/validation/test split to maintain distribution
- **Feature correlation analysis**: Identified 15 multicollinear pairs for potential dimensionality reduction

### Model Development
- **Baseline XGBoost**: Establishes performance benchmark with traditional gradient boosting
- **NeurFS Implementation**: Custom neural architecture with embedding layers for categorical features
- **Hyperparameter optimization**: Systematic tuning for optimal performance

### Robust Evaluation
- **MAE (Mean Absolute Error)**: Primary metric, robust to outliers in skewed distribution
- **RMSE**: Secondary metric for model comparison
- **Stratified validation**: Ensures model generalization across claim severity ranges

## Technologies Used

- **Python 3.x** (Google Colab environment)
- **Core Libraries**:
  - `pandas` - Data manipulation and analysis
  - `numpy` - Numerical computing
  - `scikit-learn` - Preprocessing, model evaluation, and baseline algorithms
  - `tensorflow/keras` - Neural network development and NeurFS implementation
  - `xgboost` - Gradient boosting baseline model
  - `matplotlib` - Data visualization
  - `seaborn` - Statistical data visualization
  - `scipy` - Statistical functions and tests

## Evaluation Metrics

**Primary Metric:**
- **MAE (Mean Absolute Error)**: Chosen for robustness to outliers in the highly skewed loss distribution. Provides interpretable dollar-amount prediction error.

**Secondary Metrics:**
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more heavily, useful for identifying model performance on high-severity claims
- **R-squared**: Measures proportion of variance explained by the model
- **Log-transformed metrics**: Evaluated on log-scale to assess performance on transformed target

**Why MAE is Primary:**
Given the extreme skewness (3.79) and presence of outlier claims ($121K max vs $3K mean), MAE provides a more stable and interpretable measure of model accuracy than RMSE, which can be disproportionately influenced by extreme values.

## Data Challenges and Solutions

| Challenge | Impact | Solution Implemented |
|-----------|--------|---------------------|
| **Extreme skewness** (3.79) | Non-normal distribution violates linear model assumptions | Log transformation of target variable |
| **High-cardinality categoricals** (cat116 = 326 values) | One-hot encoding would create 326+ sparse columns causing memory explosion | Neural embeddings via NeurFS |
| **Massive variability** (CV = 95.6%) | Difficult to predict with simple linear features | Neural networks to capture complex interactions |
| **Multicollinearity** (15 correlated pairs) | Redundant information in feature space | Feature selection and correlation analysis |
| **Weak linear correlations** (max 0.14) | Traditional linear models perform poorly | Deep learning for non-linear pattern recognition |
| **No frequency data** (0% zero losses) | Cannot model traditional claim frequency | Focus on severity prediction; secondary binary classification |

## Results

[Final model results will be updated upon project completion in December 2025]

### Model Performance Comparison

| Model Type | MAE (Log Scale) | RMSE | R² | Training Time |
|------------|-----------------|------|-----|---------------|
| XGBoost Baseline | [pending] | [pending] | [pending] | [pending] |
| NeurFS (Neural Network) | [pending] | [pending] | [pending] | [pending] |

### EDA Key Findings

**Completed October 2025:**
- ✅ Successfully processed 188,318 clean records with 0 missing values
- ✅ Identified extreme right skew (3.79) requiring log transformation
- ✅ Discovered weak linear correlations (max 0.14), justifying neural network approach
- ✅ Analyzed high-cardinality features (cat116 with 326 unique values)
- ✅ Confirmed dataset is severity-only (no claim frequency modeling possible)
- ✅ Validated 95.6% coefficient of variation indicating high prediction complexity

### Insights from Analysis

- **Neural networks are well-suited** for this problem due to high-dimensional categorical data and weak linear relationships
- **Embedding layers are essential** to handle high-cardinality features efficiently without memory issues
- **Log transformation is necessary** to normalize the extreme skew in claim amounts
- **Large sample size** (188K records) provides sufficient data for deep learning approaches

## Project Structure

```
insurance-claim-severity-prediction/
│
├── data/
│   ├── allstate_claims.csv         # Raw Allstate dataset (188K records)
│   └── processed/                   # Preprocessed data with log transformation
│
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis (Completed Oct 2025)
│   ├── 02_Preprocessing.ipynb      # Data cleaning and transformation
│   ├── 03_XGBoost_Baseline.ipynb  # Traditional ML baseline model
│   └── 04_NeurFS_Model.ipynb      # Neural network with embeddings
│
├── src/
│   ├── preprocessing.py            # Data preprocessing functions
│   ├── embeddings.py               # Embedding layer implementations
│   ├── neurfs_model.py             # NeurFS architecture
│   └── evaluation.py               # Model evaluation utilities
│
├── models/
│   └── saved_models/               # Trained model checkpoints
│
├── reports/
│   ├── biweekly_progress_oct17.pdf # EDA progress report
│   └── figures/                     # Visualizations and analysis charts
│
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## Installation and Setup

```bash
# Clone the repository
git clone (https://github.com/KwakyeCA)
cd Claim-Severity-and-Count-Prediction-NN-DL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Example: Load and preprocess data
import pandas as pd
from src.preprocessing import log_transform, stratified_split

# Load Allstate dataset
df = pd.read_csv('data/allstate_claims.csv')

# Apply log transformation to target
df['log_loss'] = log_transform(df['loss'])

# Create stratified split (60/20/20)
train, val, test = stratified_split(df, target='log_loss', 
                                   ratios=[0.6, 0.2, 0.2])

# Example: Train XGBoost baseline
import xgboost as xgb
from src.evaluation import calculate_mae

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=6
)
xgb_model.fit(train[features], train['log_loss'])
predictions = xgb_model.predict(val[features])
mae = calculate_mae(val['log_loss'], predictions)

# Example: Train NeurFS model with embeddings
from src.neurfs_model import build_neurfs_model
from src.embeddings import create_embedding_layers

# Build model with embedding layers for categorical features
model = build_neurfs_model(
    categorical_features=116,
    continuous_features=14,
    embedding_dims={'cat116': 50}  # 326 values → 50-dim embedding
)

model.compile(optimizer='adam', loss='mae')
model.fit(train_data, train_labels, 
         validation_data=(val_data, val_labels),
         epochs=50, batch_size=256)
```

## Future Enhancements

- [ ] **Ensemble modeling**: Combine XGBoost and NeurFS predictions for improved accuracy
- [ ] **Hyperparameter optimization**: Systematic grid search for optimal neural architecture
- [ ] **Feature importance analysis**: Extract and visualize learned embedding patterns
- [ ] **Binary classification model**: Develop high-cost vs. low-cost claim classifier for triage
- [ ] **Cross-validation**: Implement k-fold CV for more robust performance estimates
- [ ] **Model interpretability**: Apply SHAP values to explain neural network predictions
- [ ] **Production deployment**: Package model as API for real-time claim cost estimation
- [ ] **External data integration**: Incorporate additional risk factors (weather, economic indicators)

## Business Applications

This modeling framework can be applied to:
- **Actuarial Pricing**: Setting premium rates based on predicted claims
- **Risk Selection**: Identifying profitable customer segments
- **Reserve Estimation**: Calculating appropriate loss reserves
- **Portfolio Optimization**: Balancing risk across insurance book
- **Fraud Detection**: Identifying anomalous claim patterns

## References

1. McCullagh, P., & Nelder, J. A. (1989). Generalized Linear Models (2nd ed.). Chapman & Hall.
2. Frees, E. W. (2010). Regression Modeling with Actuarial and Financial Applications. Cambridge University Press.
3. Kuo, C. C., & Lu, C. J. (2017). Improving insurance claim prediction using machine learning techniques. Journal of Risk and Insurance, 84(3), 987–1012.
4. Henckaerts, R., Antonio, K., Clijsters, M., & Verbelen, R. (2021). Boosting insights in insurance tariff plans with tree-based machine learning methods. North American Actuarial Journal, 25(2), 226–247.
5. Richman, R., & Wüthrich, M. V. (2021). Deep Learning for Actuaries. SSRN.
6. Molnar, C. (2020). Interpretable Machine Learning: A Guide for Making Black Box Models Explainable.

## Author

**Cosmos Ameyaw Kwakye**  
MSc Candidate, Data Analytics Engineering | BSc Actuarial Science
Graduate Student Ambassador - Data Analytics Engineering Program
College of Engineering|Northeastern University, Vancouver

- LinkedIn: (https://linkedin.com/in/cosmos-ameyaw-kwakye-neu24dae)
- Email: successkac2020@gmail.com
- Website: (http://www.magiccna.com)

## License

**Note**: This project was developed as part of graduate coursework (Neural Networks & Deep Learning) in Data Analytics Engineering at Northeastern University (December 2025).
