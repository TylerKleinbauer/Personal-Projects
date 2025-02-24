import requests
import pandas as pd
import json


def get_and_process_snb_data(url):

    response = requests.get(url)

    data = response.json()

    all_series = []

    for series in data['timeseries']:
        header  = series['header'][0]['dimItem']
        values = series['values']

        df = pd.DataFrame(values)

        # Convert the values to numeric
        df['value'] = pd.to_numeric(df['value'])
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        all_series.append((df, header))

    processed_df=all_series[0][0][['date']].copy()

    for df, header in all_series:
        processed_df = processed_df.merge(
            df[['date', 'value']],
            on='date',
            how='outer'
        )
        processed_df = processed_df.rename(columns={'value': header})

    processed_df=processed_df.sort_values('date')

    return processed_df

def group_data_by_year(df):
    # Extract year from date column
    df['year'] = df['date'].dt.year
    
    # Group by year and take mean of numeric columns
    yearly_df = df.groupby('year').mean(numeric_only=True).reset_index()
    
    # Round all numeric columns to 5 decimal places
    numeric_cols = yearly_df.select_dtypes(include=['float64', 'int64']).columns
    yearly_df[numeric_cols] = yearly_df[numeric_cols].round(5)
    
    # Convert year back to datetime for consistency
    yearly_df['date'] = pd.to_datetime(yearly_df['year'].astype(str))
    
    # Drop the year column since we have date
    yearly_df = yearly_df.drop('year', axis=1)
    
    # Reorder columns to put date first
    cols = ['date'] + [col for col in yearly_df.columns if col != 'date']
    yearly_df = yearly_df[cols]
    
    return yearly_df

def import_snb_housing_data(url="https://data.snb.ch/api/cube/plimoincha/data/json/en"):
    try:
        # Get the data from the SNB API
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        # Create a list to store all data
        rows = []
        
        # Process each timeseries
        for series in data['timeseries']:
            # Extract header information
            property_type = series['header'][0]['dimItem']
            data_provider = series['header'][1]['dimItem']
            column_name = f"{property_type} - {data_provider}"
            
            # Add each value row with metadata
            for value in series['values']:
                rows.append({
                    'date': pd.to_datetime(value['date']),
                    'value': pd.to_numeric(value['value']),
                    'column_name': column_name
                })
        
        # Create DataFrame from all rows
        df = pd.DataFrame(rows)
        
        # Pivot to get wide format
        base_df = df.pivot(index='date', columns='column_name', values='value')
        base_df = base_df.reset_index()
        
        return base_df.sort_values('date')
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None