import mlflow
import joblib

from model_functions import train_final_model, get_model_coefficients, drop_other_targets
from data_ingestion import get_and_merge_all_data
from preprocessing_engineering_selection import prepare_data

## Note to self, pick one of these two below

def main():
    # Get and merge all data
    try:
        merged_df = get_and_merge_all_data()
    except Exception as e:
        print(f"Error getting and merging data: {e}")
        return
    
    # Preprocess data
    try:
        preprocessed_df = prepare_data(merged_df)
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return

    # Set up MLflow
    mlflow.set_experiment("final_model")
    
    with mlflow.start_run(run_name="final_model_training"):
        # Get data
        model_df = drop_other_targets(preprocessed_df, 
                                    columns={'houses_asking_price','appartments_asking_price',
                                            'appartments_transaction_price','houses_transaction_price'}, 
                                    target='houses_asking_price')
        
        # Prepare data
        X = model_df.drop(columns=['date', 'houses_asking_price'])
        y = model_df['houses_asking_price']
        
        # Train final model
        # Use the best alpha found during experimentation
        final_model = train_final_model(X, y, model_type='lasso', alpha=1.0)  # Replace with your best alpha
        
        # Save model
        model_path = "models/final_model.joblib"
        joblib.dump(final_model, model_path)
        
        # Log model
        mlflow.log_artifact(model_path)
        
        # Log parameters
        mlflow.log_params({
            "model_type": "lasso",
            "alpha": 1.0,  # Replace with your best alpha
            "features": list(X.columns),
            "target": "houses_asking_price"
        })
        
        # Get and log feature importance
        coef_df = get_model_coefficients(final_model, X)
        coef_df.to_csv("models/feature_importance.csv")
        mlflow.log_artifact("models/feature_importance.csv")
        
        print("Final model trained and saved successfully!")
        print("\nFeature Importance:")
        print(coef_df.to_string(index=False))



######################

import mlflow
import joblib
from model_functions import prepare_data, train_model_with_gs_cv, evaluate_model
from preprocessing_engineering_selection import engineer_features, generate_lagged_features, preprocess_df
from data_ingestion import get_and_merge_all_data

def train_final_model():
    # Set MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("housing_market_prediction")

    with mlflow.start_run(run_name="lasso_final_full_data"):
        # Load and preprocess data
        merged_data_df = get_and_merge_all_data()
        filtered_df = merged_data_df[(merged_data_df['date'] > '1970-01-01') & (merged_data_df['date'] < '2025-01-01')]
        df_eng = engineer_features(filtered_df)
        lagged_df = generate_lagged_features(df_eng, target_cols=['houses_asking_price'])
        preprocessed_df = preprocess_df(lagged_df, target_cols=['houses_asking_price']).ffill()
        model_df = preprocessed_df.drop(columns=['appartments_asking_price', 'appartments_transaction_price', 'houses_transaction_price'])
        model_df['regulatory_era'] = model_df['date'].dt.year.apply(lambda y: 0 if y < 2003 else 1 if y < 2011 else 2 if y < 2014 else 3 if y < 2019 else 4 if y < 2025 else 5)
        model_df['is_covid'] = model_df['date'].dt.year.isin([2020, 2021, 2022]).astype(int)

        # Prepare data (no test split—use all 40 rows)
        X = model_df.drop(columns=['date', 'houses_asking_price'])
        y = model_df['houses_asking_price']

        # Train model with best hyperparameters (e.g., alpha=0.6 from enhanced run)
        grid_search = train_model_with_gs_cv(X, y)  # Reuse your function
        best_model = grid_search.best_estimator_

        # Evaluate (e.g., CV metrics on full data)
        metrics = evaluate_model(grid_search, X, y, X, y)  # Same data for simplicity

        # Log parameters and metrics
        mlflow.log_params({
            "model_type": "lasso",
            "alpha": grid_search.best_params_['model__alpha'],
            "data_size": len(X)
        })
        mlflow.log_metrics({
            "cv_mse": metrics['cv_mse'],
            "cv_r2": metrics['cv_r2']
        })

        # Save model
        mlflow.sklearn.log_model(best_model, "final_model")
        joblib.dump(best_model, "models/final_model.joblib")

        # Log coefficients as artifact
        coef_df = get_model_coefficients(grid_search, X)
        coef_df.to_csv("models/coefficients.csv", index=False)
        mlflow.log_artifact("models/coefficients.csv")

        return best_model, metrics

if __name__ == "__main__":
    model, metrics = train_final_model()
    print(f"Final Model CV R²: {metrics['cv_r2']:.2f}")