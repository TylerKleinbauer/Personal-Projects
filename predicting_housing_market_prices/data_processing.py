import requests
import pandas as pd
import json
import io
from io import StringIO
import zipfile
import xml.etree.ElementTree as ET

########################################################################################
# Getting housing data
########################################################################################
import re

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
                    'date': pd.to_datetime(value['date'], format='%Y'),
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

def remove_sfso_data(df):
    """
    Remove Swiss Federal Statistical Office data from the DataFrame
    """
    # Get columns that don't contain 'Swiss Federal Statistical Office'
    non_sfso_cols = [col for col in df.columns 
                     if 'Swiss Federal Statistical Office' not in col or col == 'date']
    
    return df[non_sfso_cols]

def split_residental_rents(df):
    # Get column masks
    residential_mask = df.columns.str.contains('Residential property prices') | (df.columns == 'date')
    rents_mask = df.columns.str.contains('Rents') | (df.columns == 'date')
    
    return df.loc[:, residential_mask], df.loc[:, rents_mask]

def average_providers(residential_df):
    types = {
        'Residential property prices': ['Privately owned apartments', 'Single-family houses', 'Apartment buildings (residential investment property)'],
        'price_type': ['Asking price', 'Transaction price'],
    }
    
    # Initialize the averaged DataFrame with the date column
    averaged_df = pd.DataFrame({'date': residential_df['date']})
    
    # For each combination of property type and price type
    for property_type in types['Residential property prices']:
        # For each price type (asking vs transaction)
        for price_type in types['price_type']:
            # Escape special regex characters in the pattern
            pattern = re.escape(property_type)
            price_pattern = 'Asking price' if 'Asking' in price_type else 'Transaction price'
            
            # Get columns that match the property type and price type simultaneously
            matching_cols = residential_df.columns[
                residential_df.columns.str.contains(pattern, regex=True, flags=re.IGNORECASE) & 
                residential_df.columns.str.contains(price_pattern, regex=False)
            ]
            
            if len(matching_cols) > 0:
                averaged_df[f"{property_type} - {price_type}"] = residential_df[matching_cols].mean(axis=1)
    
    return averaged_df

def stack_columns(df):
    # Melt the dataframe to get long format
    id_vars = ['date']
    value_vars = [col for col in df.columns if col != 'date']
    
    melted_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars, 
                        var_name='property_price_type', value_name='value')
    
    # Split the property_price_type column
    melted_df[['property_type', 'price_type']] = melted_df['property_price_type'].str.split(' - ', n=1, expand=True)
    
    # Pivot to get asking and transaction prices as separate columns
    result_df = melted_df.pivot_table(
        index=['date', 'property_type'],
        columns='price_type',
        values='value'
    ).reset_index()
    
    # Rename columns
    result_df.columns.name = None
    result_df = result_df.rename(columns={
        'Asking price': 'asking_price',
        'Transaction price': 'transaction_price'
    })
    
    # Clean property type names
    result_df['property_type'] = result_df['property_type'].replace({
        'Apartment buildings (residential investment property)': 'Apartment buildings'
    })
    
    return result_df.sort_values(['date', 'property_type'])

def join_rents(combined_df, rents_df):
    # Rename rent columns more efficiently
    rent_cols = {col: '-'.join(col.split('-')[:2]) 
                 for col in rents_df.columns 
                 if col != 'date'}
    
    renamed_rents = rents_df.rename(columns=rent_cols)
    
    return pd.merge(combined_df, renamed_rents, on='date', how='outer')

def get_clean_housing_data(url="https://data.snb.ch/api/cube/plimoincha/data/json/en"):
    base_df = import_snb_housing_data(url)
    cleaned_df = remove_sfso_data(base_df)
    residential_df, rents_df = split_residental_rents(cleaned_df)
    averaged_df = average_providers(residential_df)
    combined_df = stack_columns(averaged_df)
    housing_df = join_rents(combined_df, rents_df)
    
    return {
        'base_df': base_df,
        'cleaned_df': cleaned_df,
        'residential_df': residential_df,
        'rents_df': rents_df,
        'averaged_df': averaged_df,
        'combined_df': combined_df,
        'housing_df': housing_df
    }

########################################################################################
# Getting and processing SNB data
########################################################################################

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

def get_and_process_snb_data_two_header_levels(url):
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

def get_and_process_mortgage_data(url="https://data.snb.ch/api/cube/bakredinausbm/data/json/en"):

    response = requests.get(url)
    data = response.json()
    
    all_series = []

    # First pass - collect all unique dates and identify all series
    for series in data['timeseries']:
        if 'Banks in Switzerland' in series['header'][0]['dimItem']:
            header = ''
            for header_value in series['header']:
                if header == '':
                    header = header_value['dimItem']
                else:
                    header += ' - ' + header_value['dimItem']

            values = series['values']
            df = pd.DataFrame(values)

            df['value'] = pd.to_numeric(df['value'])
            df['date'] = pd.to_datetime(df['date'])

            all_series.append((df, header))

    processed_df = all_series[0][0][['date']].copy()

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

def group_data_by_year_sum(df):
    # Extract year from date column
    df['year'] = df['date'].dt.year
    
    # Group by year and take sum of numeric columns
    yearly_df = df.groupby('year').sum(numeric_only=True).reset_index()
    
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

########################################################################################
# Getting and processing Demographics data
########################################################################################

def get_demographics_data(url="https://dam-api.bfs.admin.ch/hub/api/dam/assets/32229365/master"):
    try:
        # Get the data from the API
        response = requests.get(url)
        response.raise_for_status() 
        
        # Convert the response content to a string buffer
        content = StringIO(response.text)
        
        # Read the CSV from the string buffer
        raw_demographics_df = pd.read_csv(content)

    except requests.exceptions.RequestException as e:
        print(f'Request error: {e}')
    except pd.errors.EmptyDataError as e:
        print(f'CSV parsing error: {e}')
    except Exception as e:
        print(f'An error occurred: {e}')
    
    return raw_demographics_df

def long_to_wide_demographics(raw_demographics_df):
    colname = raw_demographics_df.columns[0]
    raw_demographics_df=raw_demographics_df.rename(columns={colname: 'date'})
    
    component_key = {
        'LIVB': 'Live births',
        'DTH': 'Deaths',
        'NC': 'Natural Change',
        'IMMI': 'Immigration',
        'EMI': 'Emigration',
        'NMIG': 'Net migration',
        'ACQCH': 'Acquisition of Swiss citizenship',
        'STATADJ': 'Statistical adjustments'
    }
    
    raw_demographics_df['POPULATION_CHANGE_COMPONENT'] = raw_demographics_df['POPULATION_CHANGE_COMPONENT'].map(component_key)
    
    # Using pivot_table to handle duplicate values (if any)
    demographics_df = raw_demographics_df.pivot_table(
        index='date',
        columns='POPULATION_CHANGE_COMPONENT',
        values='VALUE',
        aggfunc='first'  # or 'mean', 'sum', etc. depending on what you want to do with duplicates
    ).reset_index()
    
    
    # Convert the date to a datetime object
    demographics_df['date'] = pd.to_datetime(demographics_df['date'].astype(str) + '-01-01')
    
    return demographics_df


def get_swiss_population_data(url="https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=xml"):

    # Get the zip file
    response = requests.get(url)

    # Create a BytesIO object from the response content
    zip_data = io.BytesIO(response.content)

    # Open the zip file and create dataframe
    data = []
    with zipfile.ZipFile(zip_data) as zip_file:
        # Get the XML file name (assuming there's only one XML file)
        xml_filename = [name for name in zip_file.namelist() if name.endswith('.xml')][0]
        
        # Read the XML content
        with zip_file.open(xml_filename) as xml_file:
            # Parse the XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract data from XML
            for record in root.findall('.//record'):
                # Check if the record is for Switzerland
                country = record.find('field[@name="Country or Area"]')
                if country is not None and country.get('key') == 'CHE':
                    year = record.find('field[@name="Year"]').text
                    value = record.find('field[@name="Value"]').text
                    if year and value:
                        data.append({
                            'year': int(year),
                            'population': float(value)
                        })

    # Create pandas dataframe
    population_df = pd.DataFrame(data)
    population_df = population_df.rename(columns={'year': 'date'})
    population_df['date'] = pd.to_datetime(population_df['date'].astype(str) + '-01-01')
    
    population_df = population_df.sort_values('date')

    return population_df