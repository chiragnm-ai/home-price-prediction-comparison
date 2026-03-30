# Exploratory Data Analysis (EDA)

This report outlines the statistical distribution and feature relationships within the King County Housing dataset.

## 1. Target Variable Distribution
House prices are significantly right-skewed, indicating that while most homes are priced under $1M, there are several "luxury" outliers reaching up to $7M.

**Analysis Recommendation:** A log transformation of the price may improve Linear Regression performance.

## 2. Feature Correlation
Using a Pearson correlation matrix, we identified the following:
* **sqft_living (0.70):** Strongest positive correlation with price.
* **grade (0.67):** High construction quality significantly impacts value.
* **bathrooms (0.53):** Moderate correlation.

## 3. Visualizations
*(Note: In a local environment, these plots are generated using Seaborn and Matplotlib.)*

### Feature Heatmap
The heatmap below shows how physical attributes interact. For example, `sqft_living` is highly correlated with `bathrooms`, which may suggest multicollinearity in a standard OLS model.



### Price vs. Living Area
There is a clear linear trend between square footage and price, though the variance increases as homes get larger.
