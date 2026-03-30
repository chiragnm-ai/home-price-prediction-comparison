import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import config

def run_experiment():
    # Load data directly from public source
    df = pd.read_csv(config.DATA_PATH)
    
    X = df[config.FEATURES]
    y = df[config.TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1)
    }

    print("--- Model Performance Comparison ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        print(f"{name}:")
        print(f"  R2 Score: {r2:.4f}")
        print(f"  Mean Absolute Error: ${mae:,.2f}\n")

if __name__ == "__main__":
    run_experiment()
