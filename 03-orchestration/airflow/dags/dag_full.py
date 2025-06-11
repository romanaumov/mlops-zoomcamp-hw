#!/usr/bin/env python
# coding: utf-8

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import pickle
from pathlib import Path
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import mlflow
import os

# Default arguments for the DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'nyc_taxi_ml_pipeline',
    default_args=default_args,
    description='NYC Taxi Duration Prediction ML Pipeline',
    schedule_interval=None,  # Manual trigger
    catchup=False,
    tags=['ml', 'taxi', 'mlflow'],
)

def setup_mlflow(**context):
    """Setup MLflow tracking"""
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("nyc-taxi-experiment")
    
    # Create models directory
    models_folder = Path('/opt/airflow/models')
    models_folder.mkdir(exist_ok=True)
    
    print("MLflow setup completed")
    return "MLflow configured"

def read_dataframe_task(**context):
    """Question 3: Read March 2023 Yellow taxi data and count records"""
    # For yellow taxi data (March 2023)
    url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    
    print(f"Loading data from: {url}")
    df = pd.read_parquet(url)
    
    record_count = len(df)
    print(f"Number of records loaded: {record_count:,}")
    
    # Store data for next task
    df_path = '/opt/airflow/data/raw_data.parquet'
    os.makedirs('/opt/airflow/data', exist_ok=True)
    df.to_parquet(df_path)
    
    # Push record count to XCom for verification
    context['task_instance'].xcom_push(key='raw_record_count', value=record_count)
    
    return record_count

def prepare_data_task(**context):
    """Question 4: Data preparation and count filtered records"""
    # Load raw data
    df_path = '/opt/airflow/data/raw_data.parquet'
    df = pd.read_parquet(df_path)
    
    print("Starting data preparation...")
    
    # Data preparation (adjusted for yellow dataset)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    
    print(f"Records before filtering: {len(df):,}")
    
    # Filter duration between 1 and 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    # Convert categorical features to string
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    prepared_record_count = len(df)
    print(f"Records after preparation: {prepared_record_count:,}")
    
    # Save prepared data
    prepared_path = '/opt/airflow/data/prepared_data.parquet'
    df.to_parquet(prepared_path)
    
    # Push to XCom
    context['task_instance'].xcom_push(key='prepared_record_count', value=prepared_record_count)
    
    return prepared_record_count

def train_linear_model_task(**context):
    """Question 5: Train linear regression model and get intercept"""
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("nyc-taxi-experiment")
    
    # Load prepared data
    prepared_path = '/opt/airflow/data/prepared_data.parquet'
    df = pd.read_parquet(prepared_path)
    
    print("Training linear regression model...")
    
    with mlflow.start_run(run_name="linear_regression_model") as run:
        # Prepare features - use pickup and dropoff locations separately
        categorical = ['PULocationID', 'DOLocationID']  # Separate features
        numerical = ['trip_distance']
        
        # Create feature dictionaries
        feature_dicts = df[categorical + numerical].to_dict(orient='records')
        
        # Fit dict vectorizer
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(feature_dicts)
        
        # Target variable
        y = df['duration'].values
        
        # Train linear regression with default parameters
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        # Get and log intercept
        intercept = lr_model.intercept_
        print(f"Model intercept: {intercept:.2f}")
        
        # Log parameters and metrics
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", categorical + numerical)
        mlflow.log_metric("intercept", intercept)
        
        # Save preprocessor
        preprocessor_path = '/opt/airflow/models/preprocessor.pkl'
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(dv, f)
        
        # Save model
        model_path = '/opt/airflow/models/linear_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(lr_model, f)
        
        # Log artifacts
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
        mlflow.sklearn.log_model(lr_model, artifact_path="models_mlflow")
        
        run_id = run.info.run_id
        
        # Push to XCom
        context['task_instance'].xcom_push(key='model_intercept', value=intercept)
        context['task_instance'].xcom_push(key='linear_run_id', value=run_id)
        
        print(f"Linear regression MLflow run_id: {run_id}")
        
        return intercept

def train_xgboost_model_task(**context):
    """Train XGBoost model with original parameters"""
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("nyc-taxi-experiment")
    
    # Load prepared data
    prepared_path = '/opt/airflow/data/prepared_data.parquet'
    df = pd.read_parquet(prepared_path)
    
    print("Training XGBoost model...")
    
    with mlflow.start_run(run_name="xgboost_model") as run:
        # Create combined feature (as in original script)
        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
        
        # Prepare features
        categorical = ['PU_DO']
        numerical = ['trip_distance']
        
        feature_dicts = df[categorical + numerical].to_dict(orient='records')
        
        # Fit dict vectorizer
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(feature_dicts)
        y = df['duration'].values
        
        # Split data (80/20)
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # XGBoost training
        train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        valid_dmatrix = xgb.DMatrix(X_val, label=y_val)
        
        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:squarederror',  # Updated objective
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }
        
        mlflow.log_params(best_params)
        
        booster = xgb.train(
            params=best_params,
            dtrain=train_dmatrix,
            num_boost_round=30,
            evals=[(valid_dmatrix, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Evaluate model
        y_pred = booster.predict(valid_dmatrix)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        # Save preprocessor
        preprocessor_path = '/opt/airflow/models/xgb_preprocessor.pkl'
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(dv, f)
        
        # Log artifacts
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
        
        run_id = run.info.run_id
        
        context['task_instance'].xcom_push(key='xgb_run_id', value=run_id)
        context['task_instance'].xcom_push(key='rmse', value=rmse)
        
        print(f"XGBoost MLflow run_id: {run_id}")
        print(f"RMSE: {rmse:.4f}")
        
        return run_id

def check_model_size_task(**context):
    """Question 6: Check registered model size"""
    print("Checking model size in MLflow...")
    
    # Get the run_id from previous task
    linear_run_id = context['task_instance'].xcom_pull(key='linear_run_id', task_ids='train_linear_model')
    
    if linear_run_id:
        print(f"Linear model run_id: {linear_run_id}")
        print("Please check MLflow UI at http://localhost:5000 for model size details")
        print("Navigate to the run and check the MLmodel file for 'model_size_bytes' field")
    
    return "Check MLflow UI for model size"

def print_results_task(**context):
    """Print all results for homework questions"""
    ti = context['task_instance']
    
    # Get results from XCom
    raw_count = ti.xcom_pull(key='raw_record_count', task_ids='read_dataframe')
    prepared_count = ti.xcom_pull(key='prepared_record_count', task_ids='prepare_data')
    intercept = ti.xcom_pull(key='model_intercept', task_ids='train_linear_model')
    linear_run_id = ti.xcom_pull(key='linear_run_id', task_ids='train_linear_model')
    xgb_run_id = ti.xcom_pull(key='xgb_run_id', task_ids='train_xgboost_model')
    
    print("=" * 50)
    print("HOMEWORK RESULTS SUMMARY")
    print("=" * 50)
    print(f"Question 1: Orchestrator - Apache Airflow")
    print(f"Question 2: Version - 2.10.5")
    print(f"Question 3: Raw records loaded - {raw_count:,}")
    print(f"Question 4: Records after preparation - {prepared_count:,}")
    print(f"Question 5: Linear model intercept - {intercept:.2f}")
    print(f"Question 6: Check MLflow UI for model size")
    print(f"Linear regression run_id: {linear_run_id}")
    print(f"XGBoost run_id: {xgb_run_id}")
    print("=" * 50)
    
    return "Results printed"

# Define tasks
setup_mlflow_task = PythonOperator(
    task_id='setup_mlflow',
    python_callable=setup_mlflow,
    dag=dag,
)

read_data_task = PythonOperator(
    task_id='read_dataframe',
    python_callable=read_dataframe_task,
    dag=dag,
)

prepare_data = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data_task,
    dag=dag,
)

train_linear_model = PythonOperator(
    task_id='train_linear_model',
    python_callable=train_linear_model_task,
    dag=dag,
)

train_xgboost_model = PythonOperator(
    task_id='train_xgboost_model',
    python_callable=train_xgboost_model_task,
    dag=dag,
)

check_model_size = PythonOperator(
    task_id='check_model_size',
    python_callable=check_model_size_task,
    dag=dag,
)

print_results = PythonOperator(
    task_id='print_results',
    python_callable=print_results_task,
    dag=dag,
)

# Define task dependencies
setup_mlflow_task >> read_data_task >> prepare_data >> [train_linear_model, train_xgboost_model]
train_linear_model >> check_model_size >> print_results
train_xgboost_model >> print_results