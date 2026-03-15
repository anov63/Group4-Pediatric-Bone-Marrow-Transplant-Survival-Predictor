import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_processed_data():
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100),
        'categorical_B': np.random.randint(0, 2, 100),
        'survival_status': np.random.randint(0, 2, 100)
    })

@patch('src.evaluate_model.joblib.load')
@patch('src.evaluate_model.pd.read_csv')
def test_model_loading_and_prediction(mock_read_csv, mock_joblib_load, mock_processed_data):
    mock_read_csv.return_value = mock_processed_data
    
    mock_model = MagicMock()
    mock_model.predict.return_value = np.zeros(10)
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]] * 10)
    mock_joblib_load.return_value = mock_model
    
    from src.evaluate_model import main as eval_main
    
    with patch('src.evaluate_model.Path.exists', return_value=True):
        with patch('src.evaluate_model.plt.savefig'), patch('builtins.open'):
            eval_main()
            
    assert mock_model.predict.called
    assert mock_model.predict_proba.called
