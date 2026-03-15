import pytest
import pandas as pd
import numpy as np
from src.data_processing import optimize_memory, clean_data, handle_outliers

@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'survival_time': [100, 200, 300, 400, 500],
        'CD34kgx10d6': [1.0, 5.0, np.nan, 20.0, 1000.0],
        'categorical_feature': ['A', 'A', np.nan, 'B', 'B'],
        'numeric_feature': np.array([10.5, 20.5, 30.5, 40.5, 50.5], dtype='float64'),
        'survival_status': np.array([0, 1, 0, 1, 0], dtype='int64')
    })

def test_optimize_memory(sample_raw_data):
    df_optimized = optimize_memory(sample_raw_data.copy())
    
    assert df_optimized['numeric_feature'].dtype == 'float32'
    assert df_optimized['survival_status'].dtype == 'int32'
    assert df_optimized['survival_status'].iloc[0] == 0

def test_clean_data(sample_raw_data):
    df_cleaned = clean_data(sample_raw_data.copy())
    
    assert 'id' not in df_cleaned.columns
    assert 'survival_time' not in df_cleaned.columns
    assert df_cleaned['CD34kgx10d6'].isnull().sum() == 0
    assert df_cleaned['CD34kgx10d6'].iloc[2] == 12.5 
    assert 'categorical_feature_B' in df_cleaned.columns

def test_handle_outliers(sample_raw_data):
    np.random.seed(42)
    large_df = pd.DataFrame({
        'CD34kgx10d6': np.append(np.random.normal(10, 2, 98), [-1000, 5000])
    })
    
    df_clipped = handle_outliers(large_df.copy())
    
    assert df_clipped['CD34kgx10d6'].max() < 5000
    assert df_clipped['CD34kgx10d6'].min() > -1000
