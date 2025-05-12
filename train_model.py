import mlflow
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Daily_Training")

# 2. Load/prepare data (example)
data = pd.read_csv("data.csv")
X = data.drop("target", axis=1)
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Train and log model
with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    # Log metrics/params
    mlflow.log_metric("mse", mse)
    mlflow.log_params(model.get_params())
    mlflow.sklearn.log_model(model, "model")
    
    # Save model locally
    joblib.dump(model, "best_model.pkl")