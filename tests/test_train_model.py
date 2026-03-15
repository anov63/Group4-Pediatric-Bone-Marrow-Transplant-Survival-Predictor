import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.train_model import ModelTrainer

@pytest.fixture
def mock_processed_data():
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100),
        'categorical_B': np.random.randint(0, 2, 100),
        'survival_status': np.random.randint(0, 2, 100)
    })

@patch('src.train_model.pd.read_csv')
def test_prepare_data(mock_read_csv, mock_processed_data, tmp_path):
    mock_read_csv.return_value = mock_processed_data
    
    # We use pytest's tmp_path fixture instead of a fake string
    trainer = ModelTrainer(project_root=tmp_path)
    trainer.prepare_data()
    
    assert len(trainer.X_dev) == 90
    assert len(trainer.X_holdout) == 10
    assert len(trainer.y_dev) == 90
    assert len(trainer.y_holdout) == 10
    assert 'survival_status' not in trainer.X_dev.columns
