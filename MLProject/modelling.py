import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mlflow.tracking import MlflowClient
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name) as run:
        # Debug: Cek MLflow tracking URI
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_proba)
        
        # Log parameters
        mlflow.log_param("model_type", model_name)
        if model_name == "LightGBM":
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)
            mlflow.log_param("learning_rate", model.learning_rate)
        elif model_name == "RandomForest":
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("max_depth", model.max_depth)
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc_roc", auc_roc)
        
        # Log model
        try:
            if model_name == "LightGBM":
                mlflow.lightgbm.log_model(model, model_name, input_example=X_test.iloc[:1])
                print(f"Logged LightGBM model to artifacts/{model_name}")
            elif model_name == "RandomForest":
                mlflow.sklearn.log_model(model, model_name, input_example=X_test.iloc[:1])
                print(f"Logged RandomForest model to artifacts/{model_name}")
        except Exception as e:
            print(f"Failed to log model {model_name}: {str(e)}")
            raise
        
        # Create directory for plots
        plot_dir = "plots"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Create and save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_plot_path = os.path.join(plot_dir, f"{model_name}_cm.png")
        plt.savefig(cm_plot_path)
        plt.close()
        
        # Create and save ROC curve plot
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve - {model_name}')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        roc_plot_path = os.path.join(plot_dir, f"{model_name}_roc.png")
        plt.savefig(roc_plot_path)
        plt.close()
        
        # Log plots as artifacts
        for plot_path in [cm_plot_path, roc_plot_path]:
            try:
                mlflow.log_artifact(plot_path, artifact_path="plots")
                print(f"Logged artifact {plot_path} to artifacts/plots")
            except Exception as e:
                print(f"Failed to log artifact {plot_path}: {str(e)}")
                raise
            
        # Debug statements
        print(f"MLflow artifact root: {os.getenv('MLFLOW_ARTIFACT_ROOT', 'mlruns')}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Plot path exists: {os.path.exists(plot_path)}")
        print(f"Run ID: {run.info.run_id}")
        
        # Debug: Cek artifacts di DagsHub
        client = mlflow.tracking.MlflowClient()
        try:
            artifacts = client.list_artifacts(run.info.run_id)
            print(f"Artifacts for run_id {run.info.run_id}: {[a.path for a in artifacts]}")
        except Exception as e:
            print(f"Failed to list artifacts for run_id {run.info.run_id}: {str(e)}")
        # Cetak run_id untuk GitHub Actions
        run_id = run.info.run_id
        print(f"MLFLOW_RUN_ID={run_id}")
        
        print(f"{model_name} - Accuracy: {acc:.4f}, AUC-ROC: {auc_roc:.4f}")

def main():
    # Validasi DagsHub token
    if not os.getenv('DAGSHUB_TOKEN'):
        raise ValueError("DAGSHUB_TOKEN not set in environment")
    
    # Set DagsHub tracking
    os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/johanadis/Eksperimen_SML_JohanadiSantoso.mlflow'
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'johanadis'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('DAGSHUB_TOKEN')
    
    # Set experiment
    mlflow.set_experiment("Personality_Prediction")
    
    # Load dataset
    try:
        df = pd.read_csv('personality_dataset_preprocessing.csv')
    except Exception as e:
        print(f"Failed to read dataset: {str(e)}")
        raise
    
    X = df.drop('Personality', axis=1)
    y = df['Personality']
    print(f"Feature columns: {X.columns.tolist()}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define models
    models = [
        ("LightGBM", LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ]
    
    # Train and log each model
    for model_name, model in models:
        train_and_log_model(model, model_name, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()