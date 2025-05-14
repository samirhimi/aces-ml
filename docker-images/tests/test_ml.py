import pytest
import pandas as pd
import numpy as np
from ML import ModelTrainer, app
import os
import tempfile

@pytest.fixture
def model_trainer():
    # Create a temporary directory for test data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small test dataset
        df = pd.DataFrame({
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10),
            'target': np.random.choice([0, 1], 10)
        })
        dataset_path = os.path.join(tmpdir, 'final_dataset.csv')
        df.to_csv(dataset_path, index=False)
        
        # Set environment variables for testing
        os.environ['DATASET_PATH'] = dataset_path
        os.environ['MIN_SAMPLES_FOR_TRAINING'] = '5'
        
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

def test_prometheus_metrics_endpoint(client):
    response = client.get('/prometheus_metrics')
    assert response.status_code == 200
    assert response.content_type == 'text/plain; version=0.0.4; charset=utf-8'
