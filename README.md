# Insurance Claim Severity and Frequency Prediction using Neural Networks

## Project Overview

This project develops predictive models for insurance claim analysis using both traditional Generalized Linear Models (GLMs) and advanced Neural Network architectures. The goal is to accurately predict:
- **Claim Frequency**: The number of claims expected (count data)
- **Claim Severity**: The cost/amount of individual claims (continuous data)

These predictions are critical for actuarial pricing, risk assessment, and portfolio management in property and casualty insurance.

## Business Context

Accurate claim prediction enables insurance companies to:
- Price policies appropriately based on risk
- Allocate reserves effectively
- Identify high-risk segments
- Optimize underwriting decisions
- Improve profitability while remaining competitive

## Dataset

[Provide details about your dataset - you can update this section]
- **Source**: [Dataset source/name]
- **Size**: [Number of records]
- **Features**: [Key variables used - e.g., policyholder demographics, vehicle characteristics, coverage types, etc.]
- **Target Variables**: 
  - Claim Count (Frequency)
  - Claim Amount (Severity)

## Methodology

### 1. Generalized Linear Models (GLMs)

Traditional actuarial approach using statistical distributions appropriate for insurance data:

**For Claim Frequency:**
- Poisson Regression
- Negative Binomial Regression (for overdispersion)

**For Claim Severity:**
- Gamma Regression
- Log-Normal Regression

**Advantages:**
- Interpretable coefficients
- Industry-standard approach
- Statistical significance testing
- Regulatory acceptance

### 2. Neural Network Models

Modern machine learning approach for capturing complex non-linear relationships:

**Architecture:**
- Deep Neural Networks with multiple hidden layers
- Activation functions: ReLU, Sigmoid
- Dropout layers for regularization
- Batch normalization

**Advantages:**
- Captures complex interactions
- Non-linear pattern recognition
- Superior predictive accuracy for large datasets
- Flexible feature engineering

## Key Features

- **Data Preprocessing**: Handling missing values, outlier treatment, feature scaling
- **Feature Engineering**: Creating risk indicators, interaction terms, and derived variables
- **Model Comparison**: Benchmarking GLMs vs Neural Networks on multiple metrics
- **Hyperparameter Tuning**: Optimizing model performance through systematic search
- **Model Evaluation**: Using appropriate metrics for count and continuous data

## Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas` - Data manipulation
  - `numpy` - Numerical computing
  - `scikit-learn` - GLM implementation and preprocessing
  - `tensorflow/keras` or `pytorch` - Neural network development
  - `statsmodels` - Statistical GLM implementation
  - `matplotlib/seaborn` - Data visualization

## Evaluation Metrics

**For Claim Frequency (Count Data):**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Poisson Deviance
- AIC/BIC for model comparison

**For Claim Severity (Amount Data):**
- Mean Absolute Percentage Error (MAPE)
- RMSE
- Gamma Deviance
- R-squared

## Results

[Update this section with your actual results]

### Model Performance Comparison

| Model Type | Frequency MAE | Severity RMSE | Training Time |
|------------|---------------|---------------|---------------|
| Poisson GLM | [value] | N/A | [time] |
| Gamma GLM | N/A | [value] | [time] |
| Neural Network | [value] | [value] | [time] |

### Key Insights

- Neural networks achieved [X%] improvement in prediction accuracy over traditional GLMs
- Feature importance analysis revealed [key findings]
- Model demonstrates strong performance on [specific segments]

## Project Structure

```
claim-prediction/
│
├── data/
│   ├── raw/                  # Original datasets
│   └── processed/            # Cleaned and engineered data
│
├── notebooks/
│   ├── 01_EDA.ipynb         # Exploratory Data Analysis
│   ├── 02_GLM_Models.ipynb  # Traditional GLM implementation
│   └── 03_Neural_Networks.ipynb  # NN model development
│
├── src/
│   ├── preprocessing.py      # Data cleaning and feature engineering
│   ├── glm_models.py        # GLM implementation
│   ├── nn_models.py         # Neural network architectures
│   └── evaluation.py        # Model evaluation functions
│
├── models/
│   └── saved_models/        # Trained model files
│
├── reports/
│   └── figures/             # Visualizations and charts
│
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Installation and Setup

```bash
# Clone the repository
git clone [your-repo-url]
cd claim-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Example: Training a GLM model
from src.glm_models import train_poisson_model
model = train_poisson_model(X_train, y_train)
predictions = model.predict(X_test)

# Example: Training a Neural Network
from src.nn_models import build_nn_model
nn_model = build_nn_model(input_dim=X_train.shape[1])
nn_model.fit(X_train, y_train, epochs=50, batch_size=32)
```

## Future Enhancements

- [ ] Implement Tweedie GLM for combined frequency-severity modeling
- [ ] Add ensemble methods (XGBoost, Random Forest)
- [ ] Develop interactive dashboard for model predictions
- [ ] Incorporate external data sources (weather, economic indicators)
- [ ] Deploy model as REST API for real-time predictions

## Business Applications

This modeling framework can be applied to:
- **Actuarial Pricing**: Setting premium rates based on predicted claims
- **Risk Selection**: Identifying profitable customer segments
- **Reserve Estimation**: Calculating appropriate loss reserves
- **Portfolio Optimization**: Balancing risk across insurance book
- **Fraud Detection**: Identifying anomalous claim patterns

## References

- [Add relevant actuarial/ML papers]
- [Insurance pricing methodologies]
- [Neural network architectures for regression]

## Author

**Cosmos Ameyaw Kwakye**  
MSc Candidate, Data Analytics Engineering | BSc Actuarial Science  
Northeastern University, Vancouver

- LinkedIn: [linkedin.com/in/cosmos-ameyaw-kwakye-neu24dae](https://linkedin.com/in/cosmos-ameyaw-kwakye-neu24dae)
- Email: successkac2020@gmail.com
- Website: [www.magiccna.com](http://www.magiccna.com)

## License

[Add your preferred license - e.g., MIT License]

---

**Note**: This project was developed as part of graduate coursework in Data Analytics Engineering at Northeastern University (December 2025).
