![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/license-Academic-green.svg)
![Status](https://img.shields.io/badge/status-Complete-success.svg)

## Table of Contents
- [Project Overview](#project-overview)
- [Key Results](#key-results-summary)
- [Methodology](#methodology)
- [Installation](#installation-and-setup)
- [Usage](#usage)
- [Results](#current-project-status)
- [Contact](#contact)

---

# Insurance Claim Severity Prediction Using Neural Networks
## A Margin Tolerance Evaluation Framework Developed

## Project Overview

This project develops predictive models for insurance claim severity analysis using Generalized Linear Models (GLMs), XGBoost, and advanced Neural Network architectures with entity embeddings. The primary objective is to accurately predict **claim severity** - the cost/amount of individual insurance claims given that a claim has occurred.

**Key Focus**: Severity-only modeling using the Allstate Insurance dataset, which contains exclusively filed claims (no non-claimant records). This is critical for:
- Accurate reserve estimation
- Portfolio risk assessment  
- Large claim identification and prioritization
- Pricing optimization for renewal business

**Novel Contribution**: A **Margin Tolerance Evaluation Framework** that quantifies prediction accuracy within business-relevant dollar thresholds ($500, $1,000, $2,000, $5,000), bridging the gap between traditional metrics (MAE) and operational decision-making.

**Project Status**:  **Complete** - December 2025

**Course**: IE7615 Neural Networks and Deep Learning | Northeastern University | Fall 2025

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
**Features**: 130 total features
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

**Data Quality**: 
-  0 missing values
-  0 duplicates
-  100% clean dataset

## Key Results Summary

###  Model Performance Comparison

| Model | Test MAE ($) | Test R² | Parameters | Key Advantage |
|-------|-------------|---------|------------|---------------|
| **Ensemble NN** | **$1,132** | **0.591** | ~750K | **Best performance + interpretability** |
| XGBoost | $1,145 | 0.587 | ~3M | Strong baseline |
| Single NN | $1,149 | 0.582 | ~150K | 54% dimensionality reduction |
| Tweedie GLM | $1,322 | 0.542 | ~130 | Actuarial standard baseline |

**Key Findings:**
-  **Ensemble NN achieves best performance** ($1,132 MAE)
-  **Outperforms XGBoost** by $13/claim (1.1% improvement)
-  **14.4% improvement** over traditional GLM baseline ($190/claim savings)
-  **54% dimensionality reduction**: 1,176 sparse features → 543 dense embeddings
-  **Statistical validation**: Bootstrap CI confirms Ensemble NN and XGBoost are statistically equivalent, both significantly outperform GLM

###  Novel Contribution: Margin Tolerance Analysis

A new evaluation framework assessing **practical business accuracy**:

| Error Margin | Ensemble NN | XGBoost | Single NN | Tweedie GLM | Business Context |
|--------------|-------------|---------|-----------|-------------|------------------|
| ±$500 | **39.6%** | 39.3% | 39.0% | 28.4% | Tight accuracy requirement |
| **±$1,000** | **63.9%** | 64.2% | 63.6% | 55.3% | **Individual claim budgeting** |
| ±$2,000 | **84.6%** | 84.9% | 84.3% | 82.3% | Moderate tolerance |
| ±$5,000 | **97.5%** | 97.3% | 97.3% | 97.1% | Reserve setting threshold |

**Business Insight**: At the critical $1,000 threshold, 63.9% of predictions fall within acceptable tolerance—nearly two-thirds of claims predicted accurately for pricing decisions. This represents an 8.6 percentage point improvement over GLM (55.3%).

## Methodology

### Phase 1: Exploratory Data Analysis 

**Completed Analyses:**
- Distribution analysis revealing extreme right skew (3.79)
- Feature correlation study (max correlation = 0.14, indicating weak linear relationships)
- High-cardinality categorical identification (cat116 with 326 unique values)
- Missing value and duplicate analysis (0 issues found)
- Statistical summary and outlier detection

**Key Challenges Identified:**
1. Extreme skewness (3.79) requiring transformation
2. High-cardinality categoricals (6 features with 50+ values)
3. Weak linear correlations justifying neural network approach
4. Multicollinearity (15 correlated pairs r > 0.9)

### Phase 2: Baseline Models 

#### 2A. Traditional Actuarial Baseline: Tweedie GLM

**Why Tweedie GLM:**
- Generalizes Gamma distribution for insurance loss modeling
- Superior numerical stability for extreme skewness
- Works on original dollar scale
- Industry-standard actuarial approach

**Configuration:**
- Distribution: Tweedie (power=1.5) with log link
- Regularization: L2 (α=0.01)
- Features: All 129 features (after correlation filtering)

**Results:**
- Test MAE: **$1,322**
- Test R²: **0.542**
- Status: Baseline performance benchmark

#### 2B. Modern ML Baseline: XGBoost

**Why XGBoost:**
- Handles high-cardinality categoricals efficiently
- Robust to outliers and skewed distributions
- Provides feature importance insights
- Industry-proven performance on tabular data

**Configuration:**
- Objective: reg:squarederror
- Trees: 500
- Max depth: 6
- Learning rate: 0.05
- Subsample: 0.8
- Early stopping: patience=50

**Results:**
- Test MAE: **$1,145**
- Test R²: **0.587**
- Training time: ~15 minutes

### Phase 3: Neural Networks with Embeddings 

#### Neural Frequency-Severity (NeurFS) Framework Adaptation

**Embedding Strategy:**
- 116 embedding layers for categorical features
- Dimensionality rule: `dim = min(50, ⌈cardinality/2⌉)` (from Lim, 2024)
- Examples:
  - cat116: 326 categories → 50-dimensional embedding
  - cat110: 131 categories → 50-dimensional embedding  
  - cat1: 2 categories → 1-dimensional embedding
- **Result**: 1,176 one-hot dimensions → 543 embedded dimensions (**54% reduction**)

**Single Neural Network Architecture:**
```
Input Layer: 543 embedded dims + 14 continuous features
    ↓
Hidden Layer 1: 256 neurons (ReLU, BatchNorm, Dropout 0.3)
    ↓
Hidden Layer 2: 128 neurons (ReLU, BatchNorm, Dropout 0.2)
    ↓
Output Layer: 1 neuron (linear activation)

Total Parameters: ~150,000 trainable weights
```

**Ensemble Neural Network (5 Models):**
- 5 diverse architectures with varying depths, widths, and regularization
- Final prediction: Simple average of 5 model outputs
- Reduces variance and improves robustness

**Training Configuration:**
- Optimizer: Adam (learning_rate=0.001)
- Loss: MSE on log-transformed target
- Batch size: 256
- Early stopping: patience=10 epochs
- Hardware: Google Colab T4 GPU

**Results:**
| Model | Test MAE | Test R² |
|-------|----------|---------|
| Single NN | $1,149 | 0.582 |
| **Ensemble NN** | **$1,132** | **0.591** |

### Phase 4: SHAP Interpretability Analysis 

**SHAP Validation of Embeddings:**
- 5 of top 10 features are high-cardinality categoricals (cat80, cat101, cat100, cat79, cat81)
- Confirms embeddings successfully capture semantic relationships

**Top 5 Features by SHAP Importance:**
1. **cat80** (mean |SHAP| = 0.178) - ~18% of prediction variance, likely geographic/policy type
2. **cat101** (0.111) - Policyholder characteristic
3. **cont14** (0.077) - Coverage amount
4. **cat100** (0.067)
5. **cat79** (0.065)

**Key Insight**: Feature hierarchy aligns with actuarial theory—geographic segmentation (cat80) and coverage amount (cont14) are fundamental to insurance pricing.

### Phase 5: Ablation Studies 

| Experiment | Configuration | MAE ($) | Finding |
|------------|---------------|---------|---------|
| Ensemble Size | 1 → 3 → 5 → 7 | 1,149 → 1,138 → 1,132 → 1,131 | 5 models optimal (diminishing returns after) |
| Embedding Dims | Fixed 10 vs NeurFS vs Fixed 50 | 1,168 vs 1,132 vs 1,141 | NeurFS formula optimal |
| Network Depth | 1 → 2 → 3 layers | 1,162 → 1,149 → 1,153 | 2 layers optimal |
| Dropout Rate | 0 → 0.2 → 0.3 → 0.5 | 1,178 → 1,155 → 1,149 → 1,161 | 0.3 optimal |

## Data Preprocessing Pipeline

### Target Transformation
- **Original distribution**: Mean=$3,037, Skewness=3.79 (severe right skew)
- **Transformation**: Natural log → Skewness=0.2 (near-normal)
- **Impact**: Enabled stable model training for all approaches

### Feature Engineering
- **Categorical encoding**: Label encoding for GLM/XGBoost, Entity embeddings for NN
- **High-cardinality handling**: Neural embeddings achieving 54% dimensionality reduction
- **Correlation filtering**: Removed 3 highly correlated features (r > 0.99)
- **Final feature count**: 129 features

### Data Splitting
- **Training**: 112,991 samples (60%)
- **Validation**: 37,664 samples (20%)
- **Test**: 37,663 samples (20%)
- **Strategy**: Stratified split maintaining target distribution

## Challenges and Solutions

| Challenge | Problem | Solution | Result |
|-----------|---------|----------|--------|
| **GLM Convergence** | Gamma GLM failed due to skewness=3.79 | Switched to Tweedie GLM (power=1.5) | Successful convergence, MAE=$1,322 |
| **Multicollinearity** | 15 feature pairs with r>0.99 | Removed one from each pair | 132→129 features, no performance loss |
| **High-Cardinality Categoricals** | 1,176 one-hot dimensions | Entity embeddings with NeurFS formula | 54% reduction to 543 dims |
| **Model Variance** | Single NN variance | Ensemble of 5 diverse architectures | $17 improvement, more robust |

## Technologies Used

**Environment:**
- Python 3.9+ (Google Colab)
- TensorFlow 2.x / Keras
- GPU acceleration (Colab T4)

**Core Libraries:**
- `pandas` (2.0+) - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Preprocessing, evaluation, GLM
- `tensorflow/keras` - Neural network implementation
- `xgboost` - Gradient boosting baseline
- `matplotlib` & `seaborn` - Visualization
- `shap` - Model interpretability
- `scipy` - Statistical functions

**Modeling Frameworks:**
- `statsmodels` - Tweedie GLM
- `xgboost.XGBRegressor` - Gradient boosting
- `tensorflow.keras` - Neural network with custom embedding layers

## Project Structure

```
ClaimSeverity-and-Count-Prediction-NN-DL/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
│
├── data/
│   └── train.csv                      # Allstate dataset (188,318 claims)
│
├── notebooks/
│   ├── 01_EDA.ipynb                   # Phase 1: Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb         # Data preprocessing & transformation
│   ├── 03_Baseline_Models.ipynb       # Phase 2: Tweedie GLM & XGBoost
│   ├── 04_Neural_Network.ipynb        # Phase 3: Single NN with embeddings
│   ├── 05_Ensemble_NN.ipynb           # Phase 3: Ensemble Neural Network
│   └── 06_SHAP_Analysis.ipynb         # Phase 4: Interpretability
│
├── src/                               # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py               # Data preprocessing utilities
│   ├── embeddings.py                  # Embedding dimension calculations
│   ├── neurfs_model.py                # Neural network architecture
│   └── evaluation.py                  # Metrics and evaluation functions
│
├── models/                            # Saved trained models
│   ├── tweedie_glm.pkl                # Tweedie GLM baseline
│   ├── xgboost_model.pkl              # XGBoost baseline
│   ├── single_nn.h5                   # Single neural network
│   └── ensemble_nn/                   # Ensemble model weights
│
└── reports/
    ├── IE7615_Final_Report.pdf        # Final project report
    └── figures/                       # Visualizations
        ├── margin_tolerance_analysis.png
        ├── model_comparison.png
        ├── shap_summary.png
        └── training_curves.png
```

## Installation and Setup

### Prerequisites
```bash
Python 3.9+
pip or conda package manager
(Optional) GPU with CUDA support for faster neural network training
```

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/KwakyeCA/ClaimSeverity-and-Count-Prediction-NN-DL.git
cd ClaimSeverity-and-Count-Prediction-NN-DL

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install Jupyter for notebooks
pip install jupyter notebook
```

### Dependencies (requirements.txt)
```
pandas>=2.0.0
numpy>=1.23.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.11.0
statsmodels>=0.14.0
shap>=0.42.0
jupyter>=1.0.0
```

## Usage

### Quick Start: Running the Complete Pipeline

```python
# 1. Load and preprocess data
import pandas as pd
from src.preprocessing import preprocess_data, log_transform, stratified_split

# Load Allstate dataset
train_df = pd.read_csv('data/train.csv')

# Preprocess
train_processed = preprocess_data(train_df)
train_processed['log_loss'] = log_transform(train_processed['loss'])

# Create splits (60/20/20)
train, val, test = stratified_split(train_processed, 
                                   target='log_loss',
                                   ratios=[0.6, 0.2, 0.2],
                                   random_state=42)
```

### Example: Margin Tolerance Analysis

```python
from src.evaluation import margin_tolerance_analysis
import numpy as np

# Get predictions from all models
pred_glm = glm_model.predict(X_test)
pred_xgb = np.exp(xgb_model.predict(X_test))
pred_nn = np.exp(nn_model.predict(X_test))
pred_ensemble = np.exp(ensemble_model.predict(X_test))

# Calculate accuracy at different margins
margins = [500, 1000, 2000, 5000]
y_test_original = np.exp(y_test)

for margin in margins:
    glm_acc = margin_tolerance_analysis(y_test_original, pred_glm, margin)
    xgb_acc = margin_tolerance_analysis(y_test_original, pred_xgb, margin)
    nn_acc = margin_tolerance_analysis(y_test_original, pred_nn, margin)
    ens_acc = margin_tolerance_analysis(y_test_original, pred_ensemble, margin)
    
    print(f"\nAccuracy within ±${margin}:")
    print(f"  GLM: {glm_acc:.1f}%")
    print(f"  XGBoost: {xgb_acc:.1f}%")
    print(f"  Single NN: {nn_acc:.1f}%")
    print(f"  Ensemble NN: {ens_acc:.1f}%")
```

## Evaluation Metrics

### Primary Metric: Mean Absolute Error (MAE)

**Why MAE?**
- Robust to outliers in highly skewed distribution (skewness=3.79)
- Interpretable dollar-amount prediction error
- Directly measures average prediction error
- Less sensitive to extreme values than RMSE

### Novel Metric: Margin Tolerance Accuracy

**Why Margin Tolerance?**
- Traditional MAE lacks business context
- Answers: "What % of predictions are within acceptable error?"
- Directly actionable for pricing and reserving decisions
- Business-relevant thresholds: $500, $1K, $2K, $5K

## Business Applications

This modeling framework enables:

1. **Actuarial Pricing**: Set premium rates based on predicted claim costs
2. **Risk Management**: Identify high-risk policies for manual review
3. **Reserve Estimation**: Calculate appropriate loss reserves with confidence intervals
4. **Portfolio Optimization**: Balance risk across insurance book
5. **Operational Efficiency**: Triage claims by predicted severity

## Key Visualizations

### 1. Target Distribution
![Target Distribution](reports/figures/target_distribution.png)
*Before/after log transformation showing skewness reduction from 3.79 to 0.2*

### 2. Categorical Feature Cardinality
![Cardinality Chart](reports/figures/cardinality_chart.png)
*Distribution of unique values across 116 categorical features*

### 3. Model Performance Comparison
![Model Performance](reports/figures/model_performance.png)
*Side-by-side MAE and R² comparison - Ensemble NN achieves best performance*

### 4. Margin Tolerance Analysis
![Margin Tolerance Analysis](reports/figures/margin_tolerance_analysis.png)
*Prediction accuracy at business-critical thresholds ($500, $1K, $2K, $5K)*

### 5. Training Curves
![Training Curves](reports/figures/training_curves.png)
*Loss convergence across epochs - Train-validation gap confirms no overfitting*

## License

This project is developed as part of graduate coursework at Northeastern University (Fall 2025). 

**Academic Use**: Free to use for educational and research purposes with proper attribution.

**Commercial Use**: Please contact the author for permissions.

## Acknowledgments

- **Course**: IE7615 Neural Networks and Deep Learning
- **Institution**: Northeastern University, College of Engineering, Vancouver Campus
- **Dataset**: Allstate Insurance Company (Kaggle competition dataset)
- **Key Reference**: Lim, D.-Y. (2024). Neural Frequency-Severity Model (NeurFS framework)

## References

Lim, D. Y. (2024). A neural frequency-severity model and its application to insurance claims. arXiv preprint arXiv:2106.10770.

Henckaerts, R., Côté, M. P., Antonio, K., & Verbelen, R. (2021). Boosting insights in insurance tariff plans with tree-based machine learning methods. North American Actuarial Journal, 25(2), 255-285.

Guo, C., & Berkhahn, F. (2016). Entity embeddings of categorical variables. arXiv preprint arXiv:1604.06737.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.

Frees, E. W. (2010). Regression modeling with actuarial and financial applications. 

Jørgensen, B., & Paes De Souza, M. C. (1994). Fitting Tweedie's compound Poisson model to insurance claims data. Scandinavian Actuarial Journal, 1994(1), 69-93.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30, 4765-4774.

Pesantez-Narvaez, J., Guillen, M., & Alcañiz, M. (2019). Predicting motor insurance claims using telematics data—XGBoost versus logistic regression. Risks, 7(2), 70.

Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. Information Fusion, 81, 84-90.


## Contact

**Cosmos Ameyaw Kwakye, BSc, MIMA**  
MSc Candidate, Data Analytics Engineering | BSc Actuarial Science  
Full Member (MIMA), Institute of Mathematics and Its Applications (IMA), UK  
Graduate Student Ambassador - Data Analytics Engineering Program  
College of Engineering | Northeastern University, Vancouver, Canada

- Email: kwakye.c@northeastern.edu
- LinkedIn: [linkedin.com/in/cosmos-ameyaw-kwakye-neu24dae](https://linkedin.com/in/cosmos-ameyaw-kwakye-neu24dae)
- Website: [www.magiccna.com](http://www.magiccna.com)
- GitHub: [github.com/KwakyeCA](https://github.com/KwakyeCA)

---

**Project Timeline**: September 2025 - December 2025  
**Last Updated**: December 2025  
**Status**:  Complete

---

 **If you find this project helpful, please consider giving it a star!**
