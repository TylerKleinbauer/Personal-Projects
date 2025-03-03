import pandas as pd
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel

def generate_lagged_features(df, max_lag=3, exclude_cols=['date'], target_cols=None):
    """
    Programmatically generate lagged versions of all features except excluded columns.

    Parameters:
    - df: Wide-format DataFrame.
    - max_lag: Maximum number of lags to create (e.g., 3 years).
    - exclude_cols: Columns to skip (e.g., 'date').

    Returns:
    - DataFrame with original and lagged features.
    """
    df_lagged = df.copy()
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    for col in feature_cols:
        for lag in range(1, max_lag + 1):
            df_lagged[f'{col}_lag{lag}'] = df_lagged[col].shift(lag)
    
    # Sort columns alphabetically but keep date first and target_cols after date
    other_cols = [col for col in df_lagged.columns if col != 'date' and col not in target_cols]
    other_cols.sort()
    
    # Build final column list only with columns that exist
    final_cols = ['date'] if 'date' in df_lagged.columns else []
    final_cols.extend(target_cols)  # Add only the targets that exist
    final_cols.extend(other_cols)
    
    df_lagged = df_lagged[final_cols]
    
    return df_lagged

def drop_non_lagged_features(df, target_cols, exclude_cols=['date']):
    """
    Drop all columns that don't contain 'lag' or aren't target columns, with optional exclusions.
    
    Parameters:
    - df: DataFrame with features and targets
    - target_cols: List of target column names to preserve
    - exclude_cols: List of additional columns to preserve (default: ['date'])
    
    Returns:
    - DataFrame with only lagged features, targets, and excluded columns
    """
    # Create list of columns to keep
    cols_to_keep = []
    
    # Add excluded columns if they exist in df
    cols_to_keep.extend([col for col in exclude_cols if col in df.columns])
    
    # Add target columns if they exist in df 
    cols_to_keep.extend([col for col in target_cols if col in df.columns])
    
    # Add all columns containing 'lag'
    lag_cols = [col for col in df.columns if 'lag' in col]
    cols_to_keep.extend(lag_cols)
    
    # Return DataFrame with only kept columns
    return df[cols_to_keep]


def select_features_by_correlation(df, target_cols, threshold=0.3):
    """
    Select features with correlation above a threshold to specified targets.

    Parameters:
    - df: DataFrame with features and targets.
    - target_cols: List of target columns (e.g., price columns).
    - threshold: Minimum absolute correlation to keep a feature.

    Returns:
    - List of selected feature names.
    """
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    # Get correlations with targets
    selected_features = set()
    for target in target_cols:
        target_corrs = corr_matrix[target].drop(target_cols)  # Exclude targets from features
        high_corr_features = target_corrs[abs(target_corrs) >= threshold].index.tolist()
        selected_features.update(high_corr_features)
    
    return list(selected_features)



def select_features_with_model(df, target_col, threshold='median'):
    """
    Use XGBoost to select important features based on feature importance.

    Parameters:
    - df: DataFrame with features and target.
    - target_col: Single target column to predict.
    - threshold: Importance threshold (e.g., 'median' or float).

    Returns:
    - List of selected feature names.
    """
    X = df.drop(columns=['date'] + target_cols)  # Features
    y = df[target_col]  # Target
    
    # Drop rows with NaN in target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Train XGBoost
    model = XGBRegressor(random_state=42)
    model.fit(X, y)
    
    # Select features based on importance
    selector = SelectFromModel(model, prefit=True, threshold=threshold)
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    return selected_features