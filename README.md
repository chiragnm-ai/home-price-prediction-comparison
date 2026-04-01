# House Price Prediction: A Model Comparison Study

This repository demonstrates a comprehensive machine learning workflow to predict housing prices using the **King County House Sales dataset**. 

The primary objective is not just high accuracy, but a comparative analysis of different regression architectures to understand the trade-offs between **interpretability** and **predictive power**.

## 📊 Dataset Overview
The dataset contains 21,613 observations of home sales in King County, WA. Key features analyzed include:
* **Physical Attributes:** Square footage (`sqft_living`), bedrooms, bathrooms, and floors.
* **Quality Metrics:** `grade` (construction quality) and `condition`.
* **Location & View:** Waterfront access and view ratings.

## 🛠️ Methodology & Modeling
I implemented three distinct modeling approaches to evaluate their performance:

1. **Ordinary Least Squares (OLS) Regression:** Used as a baseline to understand feature coefficients and statistical significance.
2. **Random Forest Regressor:** An ensemble method to capture non-linear relationships and reduce variance.
3. **XGBoost:** A gradient-boosted decision tree approach optimized for high-performance predictive modeling.

## 📈 Model Performance Comparison

| Model | R-Squared ($R^2$) | MAE ($) | Best Use Case |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | 0.6146 | 154,726 | Fast baseline & feature importance |
| **Random Forest** | 0.6447 | 140,029 | Robustness against outliers |
| **XGBoost** | 0.6203 | 136,467 | Maximum predictive precision |

## 🔬 Results Analysis
* **The "Winner"**: XGBoost has the lowest MAE ($136,467). In real estate, being "less wrong" on average is often the goal. Even though its $R^2$ is slightly lower than Random Forest, its ability to minimize absolute error makes it the superior choice for a pricing engine.
* **Model Complexity vs. Gain**: Moving from a simple Linear Regression to a complex Random Forest only improved $R^2$ by about 3% (0.61 to 0.64) which shows the gain may not be worth the added complexity.
* *Interpretation*: This suggests that the current features (sqft, bedrooms, etc.) have a strong linear relationship with price, and the "unexplained" variance might be due to things not in the data (like school districts or recent renovations).
* **The MAE Reality Check**: An average error of ~$136k might seem high, but remember that Seattle house prices in this dataset can reach into the millions.

## 💡 Key Insights
* **Feature Impact:** Square footage and house grade are the strongest predictors across all models.
* **The "Complexity" Trade-off:** While XGBoost provided the lowest error, Linear Regression offered immediate transparency into how much each additional square foot increases the home value ($).

## 🚀 How to Run
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the training script: `python src/train.py`. 
