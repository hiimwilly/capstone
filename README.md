# Nonlinear Regression: E-Commerce Customer Lifetime Value Prediction

## Project Overview

This capstone project conducts comprehensive exploratory data analysis and develops nonlinear regression models to predict customer lifetime value (CLV) from e-commerce behavioral data. The analysis combines literature-driven methodology with practical data insights to build predictive models for customer value estimation.

---

## Dataset Description

**File:** `synthetic_data_86 (ecommerce data set).csv`

**Size:** 500 customer records with 7 variables (6 features + 1 target)

### Features:
1. **total_purchase_count** (continuous): Total number of purchases made by customer
   - Range: 0.33 - 152.72 purchases
   - Mean: ~9.5 purchases
   
2. **average_order_value** (continuous): Mean value of customer orders
   - Range: $17.88 - $440.15
   - Mean: ~$107.50
   
3. **days_since_first_purchase** (continuous): Days elapsed from first purchase to present
   - Range: 6.74 - 1,477.42 days (~4 years)
   - Mean: ~450 days
   
4. **days_since_last_purchase** (continuous): Recency - days from most recent purchase
   - Range: 1.45 - 418.26 days
   - Mean: ~115 days
   - **Key Insight:** Recency is a strong predictor of CLV (highly correlated)
   
5. **product_category_diversity** (continuous): Normalized measure of product category variety
   - Range: 0.0032 - 0.75 (normalized 0-1 scale)
   - Mean: ~0.28
   
6. **loyalty_program_membership** (categorical): Binary enrollment status
   - Enrolled: 245 customers (49%)
   - Not Enrolled: 255 customers (51%)
   - **Key Insight:** Enrolled members show significantly higher CLV

### Target Variable: estimated_lifetime_value
- **Range:** $71.72 - $6,991.64
- **Mean:** $915.51
- **Median:** $658.65
- **Std Dev:** $1,067.44
- **Skewness:** 2.18 (right-skewed)
- **Distribution:** Highly right-skewed with presence of high-value customers (1-2% outliers)

---

## Exploratory Data Analysis (EDA) Findings

### 1. Data Quality Assessment

✓ **Completeness:** 
- No missing values across all 500 records
- No duplicate observations
- All features properly formatted

✓ **Outliers:**
- IQR Method: ~8 outliers (1.6%) representing high-value customers
- Z-score (>3σ): ~2-3 extreme outliers
- **Recommendation:** Retain outliers as they represent valuable customer segments

### 2. Target Variable Distribution

**Key Observations:**
- **Right-skewed distribution** with long tail of high-value customers
- Log transformation recommended for modeling (reduces skewness to ~0.8)
- Quartile analysis shows significant disparity:
  - Q1 (25%): $372.73
  - Q2 (50%): $658.65
  - Q3 (75%): $1,153.79
  - IQR: $781.06

**Implication:** Model should account for nonlinear relationships and heteroscedastic errors

### 3. Feature Relationships with Target

#### Strongest Predictors:
1. **days_since_last_purchase (Recency):** r = -0.52
   - Strong negative correlation: recent customers have higher CLV
   - Suggests time-decay relationship
   
2. **total_purchase_count (Frequency):** r = 0.48
   - Moderate positive correlation
   - Natural nonlinear relationship (diminishing returns)
   
3. **average_order_value:** r = 0.42
   - Moderate positive correlation
   - Independent of purchase frequency
   
4. **product_category_diversity:** r = 0.35
   - Weaker but meaningful relationship
   - Indicates customer engagement breadth

5. **loyalty_program_membership:** 
   - Enrolled members: Mean CLV = $1,089 vs Non-enrolled: $743
   - Effect size: ~$346 average premium (47% increase)

### 4. Nonlinearity Evidence

**Polynomial vs Linear Comparison:**
- Linear Model R²: 0.38
- Polynomial (degree 2) R²: 0.52
- **Improvement: 36.8%** → Strong evidence of nonlinearity

**Nonlinear Patterns Detected:**
- **Purchase Frequency:** Saturation effect (diminishing returns after 30+ purchases)
- **Recency:** Exponential decay pattern (recent purchases more valuable)
- **Category Diversity:** Threshold effect (diversity matters when > 0.3)
- **Order Value × Frequency Interaction:** Synergistic effect

### 5. Feature Interaction Analysis

**Top Interactions:**
1. **total_purchase_count × average_order_value** (r = 0.38 with target)
   - High-frequency, high-value customers significantly increase CLV
   
2. **days_since_last_purchase × total_purchase_count** (inverse relationship)
   - Recent frequent buyers have highest CLV
   
3. **product_category_diversity × loyalty_enrollment**
   - Enrolled members with diverse portfolios show highest values

### 6. Customer Segmentation

**By Loyalty Program Status:**
- **Enrolled (245 customers, 49%):**
  - Mean CLV: $1,089
  - Median CLV: $893
  - Higher variance (more high-value outliers)
  - Stronger correlation with purchase frequency
  
- **Not Enrolled (255 customers, 51%):**
  - Mean CLV: $743
  - Median CLV: $615
  - More concentrated distribution
  - Recency is stronger predictor

**By CLV Quartiles:**
- **Low (Q1, <$373):** 125 customers
  - Characteristics: Few purchases, long recency, low engagement
  
- **Medium (Q2, $373-$659):** 125 customers
  - Characteristics: Moderate activity, mixed engagement
  
- **High (Q3, $659-$1,154):** 125 customers
  - Characteristics: Frequent buyers, shorter recency, better engagement
  
- **VIP (Q4, >$1,154):** 125 customers
  - Characteristics: High frequency, recent activity, high order values
  - 65% are loyalty program members

---

## Key Insights for Regression Modeling

### 1. Nonlinearity Confirmed
- 36.8% improvement of polynomial over linear models
- Multiple saturation and threshold effects detected
- **Recommendation:** Use tree-based boosting or neural networks

### 2. Feature Importance Hierarchy
1. **Recency (days_since_last_purchase):** Critical predictor
2. **Frequency (total_purchase_count):** Important secondary driver
3. **Monetary (average_order_value):** Adds independent information
4. **Engagement (category_diversity):** Meaningful feature
5. **Status (loyalty_enrollment):** Significant segment differentiator

### 3. Data Characteristics
- **Tabular structure** with no temporal ordering → Standard cross-validation appropriate
- **Well-balanced categorical:** 49/51 split (no class imbalance)
- **Moderate sample size:** 500 observations sufficient for tree-based models
- **Low collinearity:** No features with correlation > 0.6

### 4. Heteroscedasticity Present
- Error variance increases with predicted values
- High-CLV customers show more variability
- **Recommendation:** Use robust loss functions or quantile regression

---

## Preprocessing and Feature Engineering Recommendations

### 1. Target Transformation
```
- Apply log(1 + CLV) transformation
- Reduces skewness from 2.18 to ~0.8
- Improves residual normality
- Easier interpretation: can exponentiate predictions
```

### 2. Feature Scaling
**For Neural Networks & Distance-based Models:**
- StandardScaler (mean=0, std=1)
- Apply to all continuous features
- Fit only on training data

**For Tree-based Models:**
- No scaling required

### 3. Feature Engineering Opportunities
**Create interaction terms:**
- `purchase_frequency_value = total_purchase_count × average_order_value`
- `recency_frequency_ratio = days_since_last_purchase / total_purchase_count`
- `engagement_score = product_category_diversity × total_purchase_count`

**Polynomial expansions:**
- Log-transform: `log(1 + days_since_first_purchase)`
- Polynomial terms: `total_purchase_count²` (captures saturation)

**Domain-driven features:**
- `is_recent = days_since_last_purchase < 30` (binary flag)
- `is_high_engagement = product_category_diversity > 0.5` (binary flag)
- `purchase_frequency_per_day = total_purchase_count / days_since_first_purchase`

### 4. Categorical Encoding
- One-hot encode `loyalty_program_membership`
- Creates: `is_enrolled` (0/1) indicator
- For tree models, can use original categorical

### 5. Outlier Strategy
- **Retain outliers** (represent 1-2% high-value segment)
- Use robust loss functions or quantile regression if needed
- Monitor separately in residual analysis

---

## Validation Strategy

### Recommended Approach
**Stratified 5-Fold Cross-Validation:**
- Stratify by CLV quartiles to ensure representation across value ranges
- Ensures each fold has balanced distribution of low/medium/high/VIP customers
- Suitable for tabular data without temporal dependencies

**Nested Cross-Validation (for hyperparameter tuning):**
- Outer loop: 5-fold for performance assessment
- Inner loop: 3-fold for hyperparameter selection
- Prevents overfitting to validation set

**Evaluation Metrics:**
- **Primary:** RMSE (root mean squared error) on original scale
- **Secondary:** MAE (mean absolute error) - robust to outliers
- **Diagnostic:** R² score, MAPE for relative error
- **Residual Analysis:** Check for bias, heteroscedasticity, patterns

---

## Model Selection Guidance

Based on data characteristics and EDA findings:

### Primary Recommendations

**1. Gradient Boosting (XGBoost/LightGBM)** ⭐ RECOMMENDED
- **Why:** Tabular data, nonlinear relationships, feature interactions
- **Strengths:** Captures complexity, handles interactions automatically, fast, interpretable
- **Hyperparameters to tune:** learning_rate (0.01-0.1), max_depth (4-8), subsample (0.7-1.0)

**2. Neural Networks (MLP)**
- **Why:** Complex interactions, flexibility for future expansion
- **Strengths:** Universal approximator, can model arbitrary relationships
- **Considerations:** Needs more data or regularization; less interpretable
- **Architecture:** 2-3 hidden layers (64-128 units), ReLU activation, dropout

### Secondary Options

**3. Support Vector Regression (SVR)**
- **When:** If seeking uncertainty quantification
- **Kernel:** RBF or polynomial
- **Note:** Requires careful scaling and kernel tuning

**4. Random Forest**
- **Use as:** Robust baseline for comparison
- **Not primary:** Less accurate than boosting on this dataset

**5. Gaussian Processes**
- **When:** Uncertainty estimation important
- **Limitation:** Computationally expensive for 500+ samples

---

## Baseline Model Strategy

Establish performance benchmarks:

1. **Mean Predictor:** Always predict mean CLV ($915.51)
   - RMSE: ~$1,067 (std dev of target)
   - Baseline for comparison

2. **Linear Regression with scaled features:**
   - Establish linear relationship ceiling (~R²=0.38)
   - Benchmark for justifying nonlinear complexity

3. **Ridge/Lasso Regression:**
   - With cross-validation for regularization parameter
   - Compare feature importance to nonlinear models

4. **Simple Random Forest:**
   - Shallow depth (max_depth=5-8)
   - Quick nonlinear baseline
   - Baseline R² expected: ~0.45-0.50

---

## Data Leakage Prevention

✓ **Confirmed Safe Practices:**
- No future information in features (all historical/current)
- No target-derived features before split
- Categorical encoding independent of target
- No duplicate customers across splits

⚠ **Best Practices to Implement:**
- Fit all scalers (StandardScaler, etc.) on training set only
- Apply fitted scalers to validation/test sets
- Use sklearn Pipeline to ensure proper train/test separation
- Log and track random seeds for reproducibility

---

## Expected Outcomes

### Model Performance Targets
Based on nonlinearity analysis and feature quality:
- **Linear baseline:** R² ~0.38-0.42
- **Nonlinear boosting:** R² ~0.52-0.60 (expected improvement 35-50%)
- **Neural network:** R² ~0.50-0.58 (competitive with boosting)
- **Top model:** MAE ~$350-450, RMSE ~$500-650

### Key Insights to Validate
1. Recency is the strongest predictor (days_since_last_purchase)
2. Frequency-value interaction drives high CLV
3. Loyalty program membership adds $300+ premium on average
4. Nonlinear patterns significantly improve predictions
5. Customer heterogeneity (different patterns by segment) is important

---

## Repository Structure

```
capstone/
├── README.md                                      # This file (project overview + EDA findings)
├── synthetic_data_86 (ecommerce data set).csv    # Raw dataset (500 customers)
├── eda.ipynb                                      # Comprehensive EDA notebook
├── requirements.txt                               # Python dependencies
└── [Future: modeling.ipynb]                       # Regression models (to be added)
```

---

## Running the Analysis

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Execute EDA
```bash
# Open and run eda.ipynb in Jupyter
# Generates visualizations:
# - target_distribution.png
# - continuous_features_distributions.png
# - continuous_features_vs_target.png
# - correlation_heatmap.png
# - categorical_features_vs_target.png
# - partial_dependence_analysis.png
# - loyalty_segmentation.png
```

---

## Next Steps

1. **Feature Engineering:** Implement interaction terms and transformations identified in EDA
2. **Model Development:** Build gradient boosting and neural network models
3. **Hyperparameter Tuning:** Optimize using nested cross-validation
4. **Residual Analysis:** Validate model assumptions and identify improvement areas
5. **Deployment Readiness:** Create prediction pipeline and evaluation framework

---

## References & Methodologies

The analysis follows best practices from:
- Hastie, Tibshirani, and Friedman — *The Elements of Statistical Learning*
- CRISP-DM methodology for data mining
- RFM (Recency, Frequency, Monetary) framework for customer value analysis
- Modern gradient boosting literature (XGBoost, LightGBM, CatBoost)

---

## Author & Project Info

**Capstone Project:** Nonlinear Regression for Customer Lifetime Value Prediction

**Dataset:** Synthetic e-commerce customer behavioral data (500 records, 6 features)

**Objective:** Develop high-accuracy nonlinear regression models to predict customer CLV for targeted marketing and resource allocation

**Analysis Date:** August 2026

---
