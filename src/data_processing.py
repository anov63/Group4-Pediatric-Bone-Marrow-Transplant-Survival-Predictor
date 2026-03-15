import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from src.utils import get_logger

# Setup project root for relative imports and pathing
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
logger = get_logger(_name_)
def analyze_distributions(df: pd.DataFrame):
    """
    Analyzes skewness of numerical features to justify log transformations.
    """
    logger.info("Analyzing feature distributions...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    skewness = df[num_cols].skew().sort_values(ascending=False)
    
    # Identify highly skewed variables (Threshold > 0.75)
    high_skew = skewness[abs(skewness) > 0.75]
    for col, val in high_skew.items():
        logger.info(f"Targeting '{col}' for transformation: Skewness is {val:.2f}")
    
    return high_skew.index.tolist()

def analyze_correlations(df: pd.DataFrame, target_col: str = 'survival_status'):
    """
    Exploits the Spearman Heatmap to identify redundant features.
    """
    logger.info("Exploiting heatmap for feature redundancy...")
    corr_matrix = df.corr(method='spearman').abs()
    
    # Logic to identify pairs correlating above 0.80
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
    
    if to_drop:
        logger.info(f"Heatmap analysis identified {len(to_drop)} redundant features: {to_drop}")
    
    return corr_matrix, to_drop

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Reduces memory usage by converting types."""
    logger.info("Friend's Task: Optimizing memory usage...")
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64': df[col] = df[col].astype('float32')
        elif col_type == 'int64': df[col] = df[col].astype('int32')
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values and basic categorical encoding."""
    logger.info("Friend's Task: Cleaning data and handling missing values...")
    bad_columns = ['id', 'survival_time', 'time_to_aGvHD_III_IV', 'Relapse', 'time', 'date']
    df = df.drop(columns=[c for c in bad_columns if c in df.columns], errors='ignore')
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return optimize_memory(df)

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clips extreme values at 1st and 99th percentiles."""
    logger.info("Handling outliers via clipping...")
    target_cols = ['CD34kgx10d6', 'CD3dkgx10d8', 'WBCx10d8', 'MNCkgx10d8', 'RNCdkgx10d8']
    for col in [c for c in target_cols if c in df.columns]:
        df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
    return df
def apply_log_transformations(df: pd.DataFrame, skewed_cols: list) -> pd.DataFrame:
    """Applies log transformations to variables identified in analyze_distributions."""
    logger.info("Applying log transformations based on distribution analysis...")
    for col in skewed_cols:
        if col in df.columns and col != 'survival_status':
            new_col = f'log_{col}'
            df[new_col] = np.log(df[col].replace(0, np.nan))
            df[new_col] = df[new_col].fillna(df[new_col].median())
            df = df.drop(columns=[col])
    return df

def reduce_multicollinearity(df: pd.DataFrame, corr_matrix, target_col: str = 'survival_status') -> pd.DataFrame:
    """Drops redundant features identified in the correlation heatmap analysis."""
    logger.info("Removing redundant features to optimize model performance...")
    target_corr = corr_matrix[target_col].fillna(0).sort_values(ascending=False)
    features = target_corr.index.drop(target_col).tolist()
    
    keep = []
    for feat in features:
        if not any(corr_matrix.loc[feat, k] > 0.80 for k in keep):
            keep.append(feat)
    
    return df[keep + [target_col]]

def main():
    RAW_PATH = PROJECT_ROOT / 'data' / 'raw' / 'csv_result-bone-marrow.csv'
    PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_PATH = PROCESSED_DIR / 'final_dataset.csv'

    if not RAW_PATH.exists():
        logger.error(f"Raw data file not found at {RAW_PATH}")
        return
