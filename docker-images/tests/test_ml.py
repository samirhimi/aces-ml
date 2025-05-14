import pytest
import pandas as pd
import numpy as np
from ML import ModelTrainer, app, process_dataset
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
import io

@pytest.fixture
def test_data():
    # Create test data
    df = pd.DataFrame({
        'timestamp': ['2025-05-14 10:00:00'] * 10,
        'Abnormality class': np.random.choice([0, 1], 10),
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10)
    })
    return df

@pytest.fixture
def model_trainer():
    # Create a temporary directory for test data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small test dataset
        df = pd.DataFrame({
            'timestamp': ['2025-05-14 10:00:00'] * 10,
            'Abnormality class': np.random.choice([0, 1], 10),
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10)
        })
        dataset_path = os.path.join(tmpdir, 'test_dataset.csv')
        df.to_csv(dataset_path, index=False)
        
        # Set environment variables for testing
        os.environ['DATASET_PATH'] = dataset_path
        os.environ['MIN_SAMPLES_FOR_TRAINING'] = '5'
        os.environ['API_TOKEN'] = 'test-token'
        
        trainer = ModelTrainer()
        yield trainer

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_model_trainer_initialization(model_trainer):
    assert model_trainer is not None
    assert model_trainer.min_samples_for_training == 5
    assert model_trainer.training is False
    assert hasattr(model_trainer, 'model')
    assert hasattr(model_trainer, 'model_lock')
    assert hasattr(model_trainer, 'dataset_lock')

def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_model_status_endpoint(client):
    response = client.get('/model-status')
    assert response.status_code == 200
    assert 'status' in response.json
    assert 'is_training' in response.json

def test_metrics_endpoint_with_auth(client, test_data):
    # Test with correct auth token
    headers = {'Authorization': 'Bearer test-token'}
    
    # Test JSON data
    json_data = {
        'timestamp': '2025-05-14 10:00:00',
        'Abnormality class': 1,
        'feature1': 0.5,
        'feature2': 0.7
    }
    response = client.post('/metrics', 
                         headers=headers,
                         json=json_data)
    assert response.status_code == 200
    assert response.json['status'] == 'success'
    
    # Test CSV file upload
    csv_data = test_data.to_csv(index=False)
    data = {
        'file': (io.BytesIO(csv_data.encode()), 'test.csv')
    }
    response = client.post('/metrics',
                         headers=headers,
                         data=data,
                         content_type='multipart/form-data')
    assert response.status_code == 200
    assert response.json['status'] == 'success'

def test_metrics_endpoint_without_auth(client):
    response = client.post('/metrics', json={})
    assert response.status_code == 401
    assert 'error' in response.json

def test_invalid_metrics_data(client):
    headers = {'Authorization': 'Bearer test-token'}
    
    # Test missing required fields
    json_data = {'feature1': 0.5}
    response = client.post('/metrics',
                         headers=headers,
                         json=json_data)
    assert response.status_code == 400
    assert 'error' in response.json

def test_append_metrics(model_trainer, test_data):
    metrics = test_data.iloc[0].to_dict()
    model_trainer.append_metrics(metrics)
    assert os.path.exists(model_trainer.dataset_path)
    df = pd.read_csv(model_trainer.dataset_path)
    assert len(df) > 0

def test_should_train(model_trainer):
    # Test when no dataset exists
    os.remove(model_trainer.dataset_path)
    assert model_trainer.should_train() is False
    
    # Test with new dataset
    test_df = pd.DataFrame({
        'timestamp': ['2025-05-14 10:00:00'],
        'Abnormality class': [1],
        'feature1': [0.5]
    })
    test_df.to_csv(model_trainer.dataset_path, index=False)
    assert model_trainer.should_train() is True

@patch('joblib.dump')
def test_save_model(mock_dump, model_trainer):
    with tempfile.TemporaryDirectory() as tmpdir:
        result = model_trainer.save_model(tmpdir)
        assert result is True
        mock_dump.assert_called_once()

def test_process_dataset(model_trainer, test_data):
    # Create a test dataset file
    test_data.to_csv(model_trainer.dataset_path, index=False)
    
    # Process the dataset
    process_dataset(model_trainer.dataset_path, model_trainer)
    
    # Verify model exists
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "random_forest_model.joblib")
    assert os.path.exists(model_path)

def test_model_training(model_trainer, test_data):
    X = test_data.drop(['timestamp', 'Abnormality class'], axis=1)
    y = test_data['Abnormality class']
    X_train = X[:8]
    X_test = X[8:]
    y_train = y[:8]
    y_test = y[8:]
    
    model_trainer.train(X_train, X_test, y_train, y_test)
    assert model_trainer.training is False
    assert model_trainer.last_training_time is not None

def test_prometheus_metrics_endpoint(client):
    response = client.get('/prometheus_metrics')
    assert response.status_code == 200
    assert response.content_type == 'text/plain; version=0.0.4; charset=utf-8'
