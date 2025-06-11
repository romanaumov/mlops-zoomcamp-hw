#!/usr/bin/env python
# coding: utf-8

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow
import os

# Default arguments
default_args = {
    'owner': 'homework',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'homework_taxi_pipeline',
    default_args=default_args,
    description='Homework Questions DAG',
    schedule_interval=None,
    catchup=False,
    tags=['homework', 'taxi', 'ml'],
)

def question_3_load_data(**context):
    """Question 3: Load March 2023 Yellow taxi data"""
    print("Question 3: Loading March 2023 Yellow taxi data...")
    
    url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    df = pd.read_parquet(url)
    
    record_count = len(df)
    print(f"✅ Question 3 Answer: {record_count:,} records loaded")
    
    # Save for next step
    os.makedirs('/opt/airflow/data', exist_ok=True)
    df.to_parquet('/opt/airflow/data/yellow_taxi_march_2023.parquet')
    
    context['task_instance'].xcom_push(key='loaded_records', value=record_count)
    return record_count

def question_4_prepare_data(**context):
    """Question 4: Data preparation"""
    print("Question 4: Preparing data...")
    
    # Load data
    df = pd.read_parquet('/opt/airflow/data/yellow_taxi_march_2023.parquet')
    
    # Data preparation logic (yellow taxi dataset)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    
    # Filter duration between 1 and 60 minutes
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    # Convert categorical features
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    prepared_count = len(df)
    print(f"✅ Question 4 Answer: {prepared_count:,} records after preparation")
    
    # Save prepared data
    df.to_parquet('/opt/airflow/data/prepared_yellow_taxi.parquet')
    
    context['task_instance'].xcom_push(key='prepared_records', value=prepared_count)
    return prepared_count

def question_5_train_model(**context):
    """Question 5: Train linear regression and get intercept"""
    print("Question 5: Training linear regression model...")
    
    # Setup MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("homework-taxi-experiment")
    
    # Load prepared data
    df = pd.read_parquet('/opt/airflow/data/prepared_yellow_taxi.parquet')
    
    with mlflow.start_run(run_name="homework_linear_regression") as run:
        # Use pickup and dropoff locations separately (not combined)
        categorical = ['PULocationID', 'DOLocationID']
        numerical = ['trip_distance']
        
        # Create feature dictionaries
        feature_dicts = df[categorical + numerical].to_dict(orient='records')
        
        # Fit dict vectorizer
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(feature_dicts)
        
        # Target
        y = df['duration'].values
        
        # Train linear regression with default parameters
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        # Get intercept
        intercept = lr_model.intercept_
        print(f"✅ Question 5 Answer: Model intercept = {intercept:.2f}")
        
        # Log to MLflow
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", "PULocationID, DOLocationID, trip_distance")
        mlflow.log_metric("intercept", intercept)
        
        # Save models
        os.makedirs('/opt/airflow/models', exist_ok=True)
        
        # Save dict vectorizer
        with open('/opt/airflow/models/dict_vectorizer.pkl', 'wb') as f:
            pickle.dump(dv, f)
        
        # Save linear model
        with open('/opt/airflow/models/linear_model.pkl', 'wb') as f:
            pickle.dump(lr_model, f)
        
        # Log artifacts to MLflow
        mlflow.log_artifact('/opt/airflow/models/dict_vectorizer.pkl', artifact_path="preprocessor")
        mlflow.sklearn.log_model(lr_model, artifact_path="models_mlflow")
        
        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")
        
        context['task_instance'].xcom_push(key='intercept', value=intercept)
        context['task_instance'].xcom_push(key='run_id', value=run_id)
        
        return intercept

def question_6_check_model_size(**context):
    """Question 6: Instructions to check model size"""
    run_id = context['task_instance'].xcom_pull(key='run_id', task_ids='question_5_train_model')
    
    print("Question 6: Check model size in MLflow")
    print("=" * 50)
    print("To find the model size:")
    print("1. Open MLflow UI at http://localhost:5000")
    print("2. Go to 'homework-taxi-experiment' experiment")
    print(f"3. Find run with ID: {run_id}")
    print("4. Click on the run")
    print("5. Go to 'Artifacts' tab")
    print("6. Navigate to 'models_mlflow' folder")
    print("7. Click on 'MLmodel' file")
    print("8. Look for 'model_size_bytes' field")
    print("=" * 50)
    
    return "Check MLflow UI for model size"

def print_homework_answers(**context):
    """Print all homework answers"""
    ti = context['task_instance']
    
    loaded_count = ti.xcom_pull(key='loaded_records', task_ids='question_3_load_data')
    prepared_count = ti.xcom_pull(key='prepared_records', task_ids='question_4_prepare_data')
    intercept = ti.xcom_pull(key='intercept', task_ids='question_5_train_model')
    run_id = ti.xcom_pull(key='run_id', task_ids='question_5_train_model')
    
    print("\n" + "=" * 60)
    print("🎯 HOMEWORK ANSWERS SUMMARY")
    print("=" * 60)
    print(f"Question 1: Orchestrator Tool = Apache Airflow")
    print(f"Question 2: Version = 2.10.5")
    print(f"Question 3: Records loaded = {loaded_count:,}")
    print(f"Question 4: Records after prep = {prepared_count:,}")
    print(f"Question 5: Model intercept = {intercept:.2f}")
    print(f"Question 6: Check MLflow UI (run_id: {run_id})")
    print("=" * 60)
    print("🔗 Access MLflow UI at: http://localhost:5000")
    print("🔗 Access Airflow UI at: http://localhost:8080")
    print("=" * 60)

# Define tasks
task_q3 = PythonOperator(
    task_id='question_3_load_data',
    python_callable=question_3_load_data,
    dag=dag,
)

task_q4 = PythonOperator(
    task_id='question_4_prepare_data',
    python_callable=question_4_prepare_data,
    dag=dag,
)

task_q5 = PythonOperator(
    task_id='question_5_train_model',
    python_callable=question_5_train_model,
    dag=dag,
)

task_q6 = PythonOperator(
    task_id='question_6_check_model_size',
    python_callable=question_6_check_model_size,
    dag=dag,
)

task_summary = PythonOperator(
    task_id='print_homework_answers',
    python_callable=print_homework_answers,
    dag=dag,
)

# Set task dependencies
task_q3 >> task_q4 >> task_q5 >> task_q6 >> task_summary