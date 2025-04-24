import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from datetime import datetime
import os

def drop_other_targets(preprocessed_df, columns, target):
    """ 
    drop other targets to keep only the one for this experiment
    """
    columns.remove(target)
    model_df = preprocessed_df.drop(columns=columns)
    
    return model_df

########################################################################################
# Model functions for training and evaluating models
########################################################################################

def prepare_data(model_df, target_column='houses_asking_price', train_size=0.8):
    """Prepare the data for modeling by splitting into train and test sets."""
    X = model_df.drop(columns=['date', target_column])
    y = model_df[target_column]
    
    train_size = int(train_size * len(model_df))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    return X_train, X_test, y_train, y_test

def train_model_with_gs_cv(X_train, y_train, model_type='lasso', param_grid=None):
    """Train a model using grid search with time series cross-validation."""
    if param_grid is None:
        param_grid = {'model__alpha': list(np.arange(0.5, 1.6, 0.01).round(1))}
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(max_iter=50000) if model_type == 'lasso' else 
                Ridge() if model_type == 'ridge' else 
                ElasticNet(max_iter=50000))
    ])
    
    tscv = TimeSeriesSplit(n_splits=2)
    grid_search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    return grid_search

def evaluate_model(grid_search, X_train, y_train, X_test, y_test):
    """Evaluate the model and return metrics."""
    # Cross validation metrics
    cv_results = grid_search.cv_results_
    mean_cv_mse = -cv_results['mean_test_score'][grid_search.best_index_]
    sst = np.var(y_train) * (len(y_train) - 1)
    n_splits = TimeSeriesSplit(n_splits=2).n_splits
    test_size_per_fold = len(y_train) // (n_splits + 1)
    sse = mean_cv_mse * test_size_per_fold
    cv_r2 = 1 - (sse / sst)
    
    # Test set metrics
    y_pred = grid_search.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    return {
        'cv_mse': mean_cv_mse,
        'cv_r2': cv_r2,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'y_pred': y_pred
    }

def get_model_coefficients(grid_search, X_train):
    """Extract and format model coefficients or feature importances."""
    best_model = grid_search.best_estimator_['model']
    
    if isinstance(best_model, (Lasso, Ridge, ElasticNet)):
        coefs = best_model.coef_
        coef_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Coefficient': coefs.round(4),
            'Selected': coefs != 0
        })
        if isinstance(best_model, (Lasso, ElasticNet)):
            coef_df = coef_df[coef_df['Selected']].drop(columns='Selected')
        coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)
    else:
        importances = best_model.feature_importances_
        coef_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importances.round(4)
        }).sort_values(by='Importance', ascending=False)
    
    return coef_df

def plot_results(model_df, train_size, y_test, y_pred, best_model):
    """Create plots for model evaluation."""
    # Actual vs Predicted plot
    plt.figure(figsize=(10, 5))
    plt.plot(model_df['date'].iloc[train_size:], y_test, label='Actual', marker='o')
    plt.plot(model_df['date'].iloc[train_size:], y_pred, label='Predicted', marker='x')
    plt.legend()
    plt.title(f'Actual vs Predicted ({best_model.__class__.__name__})')
    plt.xticks(rotation=45)
    plt.savefig("plots/predictions_plot.png")
    plt.show()
    
    # Residuals plot
    plt.figure(figsize=(10, 5))
    residuals = y_test - y_pred
    plt.scatter(model_df['date'].iloc[train_size:], residuals, marker='o')
    plt.axhline(0, color='r', linestyle='--')
    plt.title(f'Residuals Over Time ({best_model.__class__.__name__})')
    plt.xticks(rotation=45)
    plt.savefig("plots/residuals_plot.png")
    plt.show()

########################################################################################
# Model functions for training the final model
########################################################################################

def train_final_model(X, y, model_type='lasso', alpha=None):
    """
    Train the final model on the complete dataset.
    Uses the best hyperparameters found during experimentation.
    
    Parameters:
    -----------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    model_type : str
        Type of model to train ('lasso', 'ridge', 'elasticnet')
    alpha : float, optional
        Regularization parameter. If None, uses the best alpha found during CV.
    """
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(max_iter=50000, alpha=alpha) if model_type == 'lasso' else 
                Ridge(alpha=alpha) if model_type == 'ridge' else 
                ElasticNet(max_iter=50000, alpha=alpha))
    ])
    
    # Fit on complete dataset
    pipeline.fit(X, y)
    
    return pipeline



