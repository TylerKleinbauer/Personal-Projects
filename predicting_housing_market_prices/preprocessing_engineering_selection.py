import pandas as pd
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor

from data_ingestion import get_vacancy_rate_data, get_new_housing_construction_data, get_urban_population_data, get_divorce_data

########################################################################################
# Data Preprocessing and Feature Engineering
########################################################################################

def impute_with_regression(df, target_col, predictor_col='date'):
    """
    Impute missing values in a target column using linear regression based on a predictor column.

    Parameters:
    - df: DataFrame with the target column containing missing values.
    - target_col: Column to impute (e.g., 'population').
    - predictor_col: Column to use as the predictor (default 'date', will extract year).

    Returns:
    - DataFrame with imputed values.
    """
    # Copy the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Extract year from date if predictor is 'date'
    if predictor_col == 'date':
        df_copy['year'] = df_copy['date'].dt.year
        predictor = 'year'
    else:
        predictor = predictor_col

    # Separate known and missing data
    known_data = df_copy.dropna(subset=[target_col])  # Rows with non-missing target values
    missing_data = df_copy[df_copy[target_col].isna()]  # Rows with missing target values

    if len(known_data) < 2:
        raise ValueError("Need at least 2 non-missing values to fit a regression model.")

    # Prepare training data
    X_train = known_data[[predictor]].values  # Predictor (e.g., year)
    y_train = known_data[target_col].values   # Target (e.g., population)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict missing values
    if not missing_data.empty:
        X_missing = missing_data[[predictor]].values
        y_pred = model.predict(X_missing)
        
        # Fill missing values in the original DataFrame
        df_copy.loc[df_copy[target_col].isna(), target_col] = y_pred

    # Drop temporary 'year' column if created
    if predictor_col == 'date':
        df_copy = df_copy.drop(columns=['year'])

    return df_copy


def preprocess_df(only_lagged_df, target_cols):
    """
    Preprocesses the input dataframe by filtering dates, imputing missing values using regression
    for population and GDP features, and backward filling remaining missing values.

    Args:
        df (pd.DataFrame): Input dataframe containing raw features

    Returns:
        pd.DataFrame: Preprocessed dataframe with imputed missing values and filtered dates

    Notes:
        - Filters data from 1985-01-01 onwards
        - Uses linear regression to impute missing values for population and GDP lag features
        - Uses backward fill for remaining missing values
    """

    # 1995 seems a good starting point because most features beging around that time.
    preprocessed_df = only_lagged_df[only_lagged_df['date'] >= '1985-01-01']

    # Population and gdp seem amenable to linear imputation
    reg_impute_cols = [
        'population_lag1',
        'population_lag2', 
        'gpd_lag1', 
        'gpd_lag2',
    ]
    for col in reg_impute_cols:
        preprocessed_df = impute_with_regression(preprocessed_df, target_col=col, predictor_col='date')
    
    # The other missing values do not seem amenable to imputing with regression so bfilling
    preprocessed_df = preprocessed_df.bfill()

    # Extract year from date column
    preprocessed_df['year'] = pd.to_datetime(preprocessed_df['date']).dt.year
    preprocessed_df = preprocessed_df.drop(columns=['date'])
    preprocessed_df = preprocessed_df.rename(columns={'year': 'date'})

    # Rorganize columns
    cols = ['date'] + [col for col in target_cols] + [col for col in preprocessed_df.columns if col != 'date' and col not in target_cols]
    preprocessed_df = preprocessed_df[cols]
    
    # Reset index after all preprocessing steps
    preprocessed_df = preprocessed_df.reset_index(drop=True)
    
    return preprocessed_df

def engineer_features(df, ma_window=3):
    """
    Engineer predictive features using lagged values to avoid leakage. Features chosen for economic
    relevance (e.g., financing costs, market heat, momentum) and demo manageability.

    Parameters:
    - df: Wide-format DataFrame with preprocessed data and renamed columns (no lags yet).
    - ma_window: Window size for moving averages (default 3 years).

    Returns:
    - DataFrame with leakage-free features.
    """
    df_eng = df.copy()
    
    # Generate lags for price columns first
    price_cols = [
        'appartments_asking_price',
        'appartments_transaction_price',
        'houses_asking_price',
        'houses_transaction_price'
    ]
    for col in price_cols:
        if col in df_eng.columns:
            for lag in range(1, ma_window + 1):  # Up to lag3 for ma3
                df_eng[f'{col}_lag{lag}'] = df_eng[col].shift(lag)
    
    # Moving Averages using lagged prices
    for col in price_cols:
        if col in df_eng.columns:
            lag_cols = [f'{col}_lag{i}' for i in range(1, ma_window + 1)]
            df_eng[f'{col}_ma{ma_window}'] = df_eng[lag_cols].mean(axis=1)
    
    # Price growth rates using lagged values
    for prop in ['appartments', 'houses']:
        df_eng[f'{prop}_asking_growth'] = (
            (df_eng[f'{prop}_asking_price_lag1'] - df_eng[f'{prop}_asking_price_lag2']) / 
            df_eng[f'{prop}_asking_price_lag2']
        )
        df_eng[f'{prop}_transaction_growth'] = (
            (df_eng[f'{prop}_transaction_price_lag1'] - df_eng[f'{prop}_transaction_price_lag2']) / 
            df_eng[f'{prop}_transaction_price_lag2']
        )
    
    # Asking Price / Transaction Price using lagged prices
    for prop in ['appartments', 'houses']:
        df_eng[f'{prop}_ask_to_trans_ratio'] = (
            df_eng[f'{prop}_asking_price_lag1'] / df_eng[f'{prop}_transaction_price_lag1']
        )
    
    # Interest rate * Mortgage Rate using lagged values
    df_eng['rate_x_mortgage'] = (
        df_eng['average_call_money_rate'].shift(1) * 
        df_eng['swiss_banks_mortgage_loans'].shift(1)
    )

    # Dropping lagged price columns
    df_eng = df_eng.drop(columns=[col for col in df_eng.columns if 'lag' in col and 'price' in col])
    
    return df_eng

def generate_lagged_features(df, max_lag=2, exclude_cols=['date'], target_cols=None):
    """
    Programmatically generate lagged versions of all features except excluded columns.

    Parameters:
    - df: Wide-format DataFrame.
    - max_lag: Maximum number of lags to create (default 2 years for manageable model size).
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


def prepare_data(merged_data_df):
    """
    Preprocesses the merged data dataframe by filtering dates, renaming columns, and engineering features.
    
    Args:
        merged_data_df (pd.DataFrame): The merged data dataframe.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    # First date filter. Keeping rows back to 1970 to include more data when shifting DF.
    filtered_df = merged_data_df[(merged_data_df['date'] > '1970-01-01') & (merged_data_df['date'] < '2025-01-01')]

    # Rename columns
    filtered_df = filtered_df.rename(columns={
        'Privately owned apartments - Asking price': 'appartments_asking_price',
        'Privately owned apartments - Transaction price': 'appartments_transaction_price',
        'Single-family houses - Asking price': 'houses_asking_price',
        'Single-family houses - Transaction price': 'houses_transaction_price',
        'Rents - Industrial and commercial space ': 'rents_industrial_commercial',
        'Rents - Office space ': 'rents_office', 
        'Rents - Rental housing units ': 'rents_houses',
        'Rents - Retail space ': 'rents_retail',
        'Acquisition of Swiss citizenship': 'acquisition_ch_citizenship',
        'Natural Change': 'pop_natural_change',
        'Net migration': 'ch_net_migration',
        'Switzerland - CHF - Call money rate (Tomorrow next) - 1 day': 'average_call_money_rate',
        'Origination - Relevant foreign currency positions': 'money_origin_foreign_currency_position',
        'Origination - Securities portfolio': 'money_origin_securities_portfolio',
        'Origination - Money market transactions': 'money_origin_money_market_transactions',
        'Origination - Other': 'money_origin_other',
        'Origination - Monetary base': 'money_origin_monetary_base',
        'Utilisation - Banknotes in circulation': 'banknotes_in_circulation',
        'Utilisation - Sight deposit accounts of domestic banks': 'sight_deposits_banks',
        'Level - Currency in circulation': 'currency_in_circulation',
        'Level - Deposits in transaction accounts': 'deposits_in_transaction_accounts',
        'Level - Monetary aggregate M3': 'M3_supply',
        'Banks in Switzerland - Total domestic and foreign - Total loans - Utilisation': 'swiss_banks_total_loans',
        'Banks in Switzerland - Total domestic and foreign - Mortgage loans - Utilisation': 'swiss_banks_mortgage_loans',
        'Value, in CHF millions - Gross domestic product': 'gpd',
        'SFSO - Inflation according to the national consumer price index': 'inflation'
    })

    # Engineer features
    df_eng = engineer_features(filtered_df)

    # Get target collumn
    target_cols = [col for col in [
            'appartments_asking_price',
            'appartments_transaction_price',
            'houses_asking_price',
            'houses_transaction_price'
        ] if col in df_eng.columns]  # Only include targets that exist in df

    # Get columns to exclude
    exclude_cols = [col for col in [
        'date', 
        'appartments_asking_price_ma3', 
        'appartments_transaction_price_ma3',	
        'houses_asking_price_ma3',	
        'houses_transaction_price_ma3',	
        'appartments_asking_growth',	
        'appartments_transaction_growth',	
        'houses_asking_growth',
        'houses_transaction_growth',
        'appartments_ask_to_trans_ratio',
        'houses_ask_to_trans_ratio',
        'rate_x_mortgage'] if col in df_eng.columns]

    # Generate laggs excluding 
    lagged_df = generate_lagged_features(df=df_eng, target_cols=target_cols, exclude_cols=exclude_cols)
    only_lagged_df = drop_non_lagged_features(df=lagged_df, target_cols=target_cols, exclude_cols=exclude_cols)

    target_cols = [col for col in [
            'appartments_asking_price',
            'appartments_transaction_price',
            'houses_asking_price',
            'houses_transaction_price'
        ] if col in only_lagged_df.columns]  # Only include targets that exist in df

    
    preprocessed_df = preprocess_df(only_lagged_df, target_cols=target_cols)

    preprocessed_df = preprocessed_df.ffill()

    return preprocessed_df


def get_and_merge_new_data(merged_data_df):
    """ 
    Adding Vacancy rates, new housing constructions, urban population, divorces, regulatory era and covid years.
    """

    # Getting new data
    vacancy_rate_df = get_vacancy_rate_data()
    housing_construction_df = get_new_housing_construction_data()
    urban_population_df = get_urban_population_data()
    divorces_df = get_divorce_data()
    
    # Start with the original dataframe
    enriched_df = merged_data_df.copy()
    
    # Merge each dataframe one by one
    dfs = [
        (vacancy_rate_df, "Vacancy rates DataFrame"),
        (housing_construction_df, "Housing Construction DataFrame"),
        (urban_population_df, "Urban Population DataFrame"),
        (divorces_df, "Divorces DataFrame")
    ]
    
    # Merge each DF with the enriched_df
    for df, name in dfs:
        # Ensure date formats are compatible
        if df['date'].dtype != enriched_df['date'].dtype:
            df['date'] = pd.to_datetime(df['date'])
            
        enriched_df = pd.merge(
            left=enriched_df,
            right=df,
            how='left',
            on='date'
        )
    
    return enriched_df

def add_covid_regulations(preprocessed_df):
    """ 
    Adding Covid years and regulatory eras to the DF.
    """

    preprocessed_df['date'] = pd.to_datetime(preprocessed_df['date'], format= '%Y')

    # Define regulatory eras based on timeline
    def assign_regulatory_era(year):
        if year < 2003:  # Pre-Basel II, informal customs
            return 0
        elif 2003 <= year < 2011:  # Basel II, early formalization
            return 1
        elif 2011 <= year < 2014:  # SBA Guidelines (2011)
            return 2
        elif 2014 <= year < 2019:  # Post-CCyB, stricter equity (2014)
            return 3
        elif 2019 <= year < 2025:  # Investment property rules (2019)
            return 4
        elif year >= 2025:  # Basel III (2025)
            return 5
        return None  # Fallback, should not happen with your data

    # Apply to model_df
    preprocessed_df['regulatory_era'] = preprocessed_df['date'].dt.year.apply(assign_regulatory_era)

    # COVID Feature
    preprocessed_df['is_covid'] = preprocessed_df['date'].dt.year.isin([2020, 2021, 2022]).astype(int)

    preprocessed_df['date'] = preprocessed_df['date'].dt.year
    
    return preprocessed_df


########################################################################################
# Feature Selection
########################################################################################

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
    X = df.drop(columns=['date'] + target_col)  # Features
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

def select_best_lag_per_feature(df, target_cols, threshold=0.5):
    """
    Select the best lag feature for each base feature based on correlation with target columns.
    
    For each feature that has lag variants (e.g., price_lag1, price_lag2, etc.), selects the lag
    that has the highest absolute correlation with each target, if it meets the threshold.

    Parameters:
    - df: DataFrame containing features and targets
    - target_cols: List of target column names to compute correlations against
    - threshold: Minimum absolute correlation required to select a lag feature (default: 0.5)

    Returns:
    - List of selected lag feature names that have the highest correlation with targets
    """
    corr_matrix = df.corr()
    selected_features = set()
    
    # Group features by base name (before '_lag')
    base_features = set(col.split('_lag')[0] for col in df.columns if '_lag' in col)
    
    for target in target_cols:
        for base in base_features:
            lag_cols = [col for col in df.columns if col.startswith(base + '_lag')]
            if lag_cols:
                # Pick lag with highest absolute correlation
                corrs = {col: abs(corr_matrix.loc[col, target]) for col in lag_cols}
                best_lag = max(corrs, key=corrs.get)
                if corrs[best_lag] >= threshold:
                    selected_features.add(best_lag)
    
    return list(selected_features)

def select_features_with_rfe(df, target_col, n_features=10):
    """
    Select features using Recursive Feature Elimination (RFE) with XGBoost.
    
    Uses RFE to iteratively remove the least important features until reaching
    the desired number of features. XGBoost is used as the estimator to determine
    feature importance at each step.

    Parameters:
    - df: DataFrame containing features and target
    - target_col: Name of the target column to predict
    - n_features: Number of features to select (default: 10)

    Returns:
    - List of selected feature names
    """
    X = df.drop(columns=['date', target_col])
    y = df[target_col]
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    model = XGBRegressor(random_state=42)
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(X, y)
    
    return X.columns[rfe.support_].tolist()


def reduce_collinearity(df, target_cols, vif_threshold=10):
    """
    Reduce multicollinearity in features by iteratively removing features with high VIF values.
    
    Uses Variance Inflation Factor (VIF) to identify and remove features that have high
    multicollinearity with other features. Features are removed one at a time until all
    remaining features have VIF values below the threshold.

    Parameters:
    - df: DataFrame containing the features
    - vif_threshold: Maximum allowed VIF value (default: 10). Features with VIF above
                    this threshold will be removed.

    Returns:
    - List of feature names that remain after removing high VIF features
    """
    X = df.drop(columns=['date'] + target_cols).dropna()
    features = X.columns.tolist()
    
    while True:
        vif_data = pd.DataFrame()
        vif_data['feature'] = features
        vif_data['VIF'] = [variance_inflation_factor(X[features].values, i) 
                          for i in range(len(features))]
        
        max_vif = vif_data['VIF'].max()
        if max_vif <= vif_threshold:
            break
        
        # Drop feature with highest VIF
        drop_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'feature']
        features.remove(drop_feature)
    
    return features

