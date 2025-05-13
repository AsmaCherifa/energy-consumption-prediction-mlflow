import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib  
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from sklearn.metrics import mean_absolute_percentage_error, explained_variance_score, max_error, confusion_matrix, classification_report
import mlflow
from mlflow.tracking import MlflowClient
import time
import os
from datetime import datetime

# STEP 1 : Load the dataset
data = pd.read_csv('energy_dataset.csv')

# STEP 2 : Preprocessing the dataset
label_encoder = LabelEncoder()
    #Label Encoding (For Categorical Data → Numbers)
data['Machine'] = label_encoder.fit_transform(data['Machine'])

    # Drop the 'Date' column as it's not useful for prediction
data.drop(columns=['Date'], inplace=True)

    # Split the dataset into features (X) and target (y)
X = data.drop(columns=['Energy Consumed'])
y = data['Energy Consumed']

    # Scaling the features --> To handle different feature ranges
    #Feature   Before Scaling	  After Scaling
    #Temp	   25	              -0.65
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 3: Train-Test Split
    # Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


try:
    mlflow.end_run()  
except:
    pass
mlflow.set_tracking_uri('http://localhost:5000')
model_registry_name = "energy_consumption_model"

# Your original models dictionary
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(),
    'KNN': KNeighborsRegressor(),
    'SVR': SVR()
}

results = []
best_mse = float('inf')
best_model = None
best_model_name = None
best_run_id = None

def log_model_run(name, model):
    """Encapsulated model training and logging logic"""
    with mlflow.start_run(run_name=name, nested=True) as run:
        # Training
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        # Prediction
        start = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start
        
        # Metrics
        metrics = {
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred),
            'Explained Variance': explained_variance_score(y_test, y_pred),
            'Max Error': max_error(y_test, y_pred),
            'Training Time': train_time,
            'Prediction Time': pred_time
        }
        
        # Logging
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name.lower().replace(" ", "_"))
        
        return metrics, run.info.run_id
    
# Main execution
with mlflow.start_run(run_name="Model Comparison") as parent_run:
    for name, model in models.items():
        try:
            metrics, run_id = log_model_run(name, model)
            results.append({'Model': name, **metrics})
            
            if metrics['MSE'] < best_mse:
                best_mse = metrics['MSE']
                best_model = model
                best_model_name = name
                best_run_id = run_id
                print(f"New best: {name} (MSE: {best_mse:.4f})")
                
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            mlflow.end_run()  # Ensure failed run is closed


client = MlflowClient()
model_backup_dir = "backup_models"
os.makedirs(model_backup_dir, exist_ok=True)

# Backup current production model
try:
    # Get the current production model version
    versions = client.get_latest_versions(model_registry_name, stages=["Production"])
    if versions:
        prod_version = versions[0]
        prod_model_uri = f"models:/{model_registry_name}/{prod_version.version}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(model_backup_dir, f"{model_registry_name}_v{prod_version.version}_{timestamp}")
        os.makedirs(backup_path, exist_ok=True)

        # Download and store model files
        mlflow.artifacts.download_artifacts(prod_model_uri, dst_path=backup_path)
        print(f"📦 Backed up current Production model v{prod_version.version} to {backup_path}")
    else:
        print("ℹ️ No Production model to backup.")
except Exception as e:
    print(f"❌ Failed to backup current Production model: {e}")

    # Register and promote best model
if best_run_id:
    try:
        model_uri = f"runs:/{best_run_id}/{best_model_name.lower().replace(' ', '_')}"
        mv = mlflow.register_model(model_uri, model_registry_name)
        
        client.transition_model_version_stage(
            name=model_registry_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"\n🚀 Promoted {best_model_name} v{mv.version} to Production")
    except Exception as e:
        print(f"\n❌ Model promotion failed: {e}")

    # Save comparison
    comparison_df = pd.DataFrame(results).sort_values('MSE')
    comparison_df.to_csv("comparison.csv", index=False)
    mlflow.log_artifact("comparison.csv")
    print("\nModel Comparison:\n", comparison_df)

# Save final artifacts (original functionality)
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'encoder.pkl')