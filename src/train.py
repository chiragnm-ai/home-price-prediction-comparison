import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import config

def run_experiment():
    # 1. Load Data
    df = pd.read_csv(config.DATA_PATH)
    
    # 2. Select Features & Target
    X = df[config.FEATURES]
    y = df[config.TARGET]
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )

    # 4. Define Models to Compare
    models = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Random Forest (Ensemble)": RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE),
        "XGBoost (Gradient Boosting)": XGBRegressor(n_estimators=100, learning_rate=0.1)
    }

    # 5. Execute Comparison
    print("--- House Price Prediction Model Comparison ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        print(f"{name}:")
        print(f"  R2 Score (Variance Explained): {r2_score(y_test, preds):.4f}")
        print(f"  MAE (Avg Error): ${mean_absolute_error(y_test, preds):,.2f}\n")

if __name__ == "__main__":
    run_experiment()
