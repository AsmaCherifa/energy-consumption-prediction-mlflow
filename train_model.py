import mlflow
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

try:
    data = pd.read_csv("energy_dataset.csv")
    print("✅ Data loaded successfully")
    print("Columns:", data.columns.tolist())
    
    if "Energy Consumed" not in data.columns:
        raise ValueError("Column 'Energy Consumed' not found in data")
        
    X = data.drop(["Energy Consumed", "Date"], axis=1)
    y = data["Energy Consumed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except Exception as e:
    print(f"❌ Data processing failed: {e}")
    raise

# 3. Training with enhanced logging
try:
    with mlflow.start_run():
        print("🚀 Starting training...")
        
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42  # Added for reproducibility
        )
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"📊 MSE: {mse:.4f}")
        
        # Enhanced logging
        mlflow.log_metric("mse", mse)
        mlflow.log_params({
            "n_estimators": 100,
            "model_type": "RandomForest",
            "data_version": "1.0"
        })
        
        # Model saving
        mlflow.sklearn.log_model(model, "model")
        joblib.dump(model, "best_model.pkl")
        print("💾 Model saved to best_model.pkl")
        
except Exception as e:
    print(f"❌ Training failed: {e}")
    raise

print("✨ Pipeline completed successfully!")