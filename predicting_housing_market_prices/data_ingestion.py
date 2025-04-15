import requests
import pandas as pd
import json
import io
from io import StringIO, BytesIO
import zipfile
import xml.etree.ElementTree as ET
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime

########################################################################################
# Getting and processing housing data
########################################################################################
import re

def import_snb_housing_data(url="https://data.snb.ch/api/cube/plimoincha/data/json/en"):
    """
    Fetches and processes housing market data from the Swiss National Bank (SNB) API.

    Parameters
    ----------
    url : str, optional
        The SNB API endpoint URL for housing market data.
        Default is "https://data.snb.ch/api/cube/plimoincha/data/json/en"

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the processed housing data with:
        - Rows indexed by date
        - Columns representing different property types and data providers
        - Values representing property prices/indices

    Raises
    ------
    requests.exceptions.RequestException
        If there's an error fetching data from the API
    json.JSONDecodeError
        If there's an error parsing the JSON response
    Exception
        For other processing errors

    Notes
    -----
    The function:
    1. Fetches JSON data from the SNB API
    2. Processes each timeseries into rows with date, value, and metadata
    3. Pivots the data into a wide format with dates as index
    4. Returns sorted data by date
    """
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
    Filters out data from the Swiss Federal Statistical Office from the housing DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing housing market data with various data providers

    Returns
    -------
    pandas.DataFrame
        DataFrame with the same structure but excluding columns containing 
        'Swiss Federal Statistical Office' in their names, except for the 'date' column

    Notes
    -----
    This function is useful for removing potentially inconsistent or differently
    scaled data from the SFSO while retaining other data providers
    """
    # Get columns that don't contain 'Swiss Federal Statistical Office'
    non_sfso_cols = [col for col in df.columns 
                     if 'Swiss Federal Statistical Office' not in col or col == 'date']
    
    return df[non_sfso_cols]

def split_residental_rents(df):
    """
    Splits the housing data DataFrame into separate residential prices and rents DataFrames.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing both residential property prices and rents data

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        A tuple containing:
        - First DataFrame with residential property price columns and date
        - Second DataFrame with rent-related columns and date

    Notes
    -----
    Both resulting DataFrames maintain the 'date' column and their respective
    time series data
    """
    
    # Get column masks
    residential_mask = df.columns.str.contains('Residential property prices') | (df.columns == 'date')
    rents_mask = df.columns.str.contains('Rents') | (df.columns == 'date')
    
    return df.loc[:, residential_mask], df.loc[:, rents_mask]

def average_providers(residential_df):
    """
    Calculates average prices across different data providers for each property 
    and price type combination.

    Parameters
    ----------
    residential_df : pandas.DataFrame
        DataFrame containing residential property prices from various providers

    Returns
    -------
    pandas.DataFrame
        DataFrame with averaged prices containing:
        - date column
        - Columns for each combination of:
            * Property types (Apartments, Single-family houses, Apartment buildings)
            * Price types (Asking price, Transaction price)
        
    Notes
    -----
    The function:
    1. Processes three property types: Privately owned apartments, Single-family houses,
       and Apartment buildings
    2. Handles both asking and transaction prices
    3. Returns averages across providers for each property-price type combination
    """
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
    """
    Restructures the wide-format housing data into a long format while preserving 
    non-price columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide-format DataFrame with columns for different property types and their 
        asking/transaction prices

    Returns
    -------
    pandas.DataFrame
        Restructured DataFrame containing:
        - date
        - property_type (simplified names)
        - asking_price
        - transaction_price
        - All other original columns

    Notes
    -----
    The function:
    1. Preserves all non-price columns
    2. Separates property type and price type information
    3. Creates separate columns for asking and transaction prices
    4. Simplifies property type names (e.g., shortens 'Apartment buildings' name)
    5. Returns data sorted by date and property type
    """
    # Define the specific columns to stack
    property_price_cols = [
        'Privately owned apartments - Asking price',
        'Privately owned apartments - Transaction price',
        'Single-family houses - Asking price',
        'Single-family houses - Transaction price',
        'Apartment buildings (residential investment property) - Transaction price'
    ]
    
    # Filter columns that exist in the DataFrame
    value_vars = [col for col in property_price_cols if col in df.columns]
    
    # Melt only the specified price columns
    melted_df = pd.melt(
        df,
        id_vars=[col for col in df.columns if col not in value_vars],  # Keep all other columns as id_vars
        value_vars=value_vars,
        var_name='property_price_type',
        value_name='value'
    )
    
    # Split the property_price_type column into property_type and price_type
    melted_df[['property_type', 'price_type']] = melted_df['property_price_type'].str.split(' - ', n=1, expand=True)
    
    # Pivot to get asking_price and transaction_price as separate columns
    pivoted_df = melted_df.pivot_table(
        index=[col for col in melted_df.columns if col not in ['property_price_type', 'price_type', 'value']],
        columns='price_type',
        values='value'
    ).reset_index()
    
    # Rename columns
    pivoted_df.columns.name = None
    pivoted_df = pivoted_df.rename(columns={
        'Asking price': 'asking_price',
        'Transaction price': 'transaction_price'
    })
    
    # Clean property type names
    pivoted_df['property_type'] = pivoted_df['property_type'].replace({
        'Apartment buildings (residential investment property)': 'Apartment buildings'
    })
    
    # Ensure all original non-price columns are retained
    return pivoted_df.sort_values(['date', 'property_type'])

def join_rents(combined_df, rents_df):
    """
    Merges the combined housing prices DataFrame with the rents DataFrame.

    Parameters
    ----------
    combined_df : pandas.DataFrame
        DataFrame containing combined housing price data
    rents_df : pandas.DataFrame
        DataFrame containing rent data for different property types

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame containing both housing prices and rents data, joined on date

    Notes
    -----
    The function:
    1. Simplifies rent column names
    2. Performs an outer join on the date column
    3. Preserves all data from both DataFrames
    """
    # Rename rent columns more efficiently
    rent_cols = {col: '-'.join(col.split('-')[:2]) 
                 for col in rents_df.columns 
                 if col != 'date'}
    
    renamed_rents = rents_df.rename(columns=rent_cols)
    
    return pd.merge(combined_df, renamed_rents, on='date', how='outer')

def get_clean_housing_data(url="https://data.snb.ch/api/cube/plimoincha/data/json/en"):
    """
    Orchestrates the complete data processing pipeline for SNB housing market data.

    Parameters
    ----------
    url : str, optional
        The SNB API endpoint URL for housing market data.
        Default is "https://data.snb.ch/api/cube/plimoincha/data/json/en"

    Returns
    -------
    dict
        Dictionary containing DataFrames at each stage of processing:
        - 'base_df': Raw data from SNB
        - 'cleaned_df': Data with SFSO entries removed
        - 'residential_df': Residential property prices only
        - 'rents_df': Rent data only
        - 'averaged_df': Averaged prices across providers
        - 'housing_df': Final processed housing data

    Notes
    -----
    This function provides access to intermediate DataFrames for debugging
    and analysis purposes while delivering the final processed dataset
    """
    base_df = import_snb_housing_data(url)
    cleaned_df = remove_sfso_data(base_df)
    residential_df, rents_df = split_residental_rents(cleaned_df)
    averaged_df = average_providers(residential_df)
    housing_df = join_rents(averaged_df, rents_df)
    
    return {
        'base_df': base_df,
        'cleaned_df': cleaned_df,
        'residential_df': residential_df,
        'rents_df': rents_df,
        'averaged_df': averaged_df,
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

########################################################################################
# Getting additional data
########################################################################################

def get_vacancy_rate_data():
    # Get the excel file
    response = requests.get(url=r"https://dam-api.bfs.admin.ch/hub/api/dam/assets/32406434/master")

    # Convert response content to bytes buffer
    content = BytesIO(response.content)

    # Read excel file from bytes buffer
    df = pd.read_excel(content, header=3)

    def find_first_na(df):
        # Get first column
        first_col = df.iloc[7:,0]
        
        # Find first NaN index
        first_na_idx = first_col.isna().idxmax()
        
        return first_na_idx

    i = find_first_na(df)

    # Get just the two columns we want
    df_1 = df.iloc[7:i, [0,-1]]

    # Clean the year column by extracting just the year numbers
    df_1.iloc[:,0] = df_1.iloc[:,0].astype(str).str.extract('(\d{4})')

    # Convert vacancy_rate to float
    df_1.iloc[:,1] = pd.to_numeric(df_1.iloc[:,1], errors='coerce')

    # Rename columns for clarity
    df_1.columns = ['date', 'vacancy_rate']

    # Fill NaN values in vacancy_rate column with its mean
    df_1['vacancy_rate'] = df_1['vacancy_rate'].infer_objects(copy=False).interpolate(method='linear')
    
    # Ensure year column is datetime
    df_1['date'] = pd.to_datetime(df_1['date'].astype(str)+'-01-01', format='%Y-%m-%d', errors='coerce')
    
    # Group by date and take the mean of duplicate dates
    df_1 = df_1.groupby('date', as_index=False).mean()
    
    return df_1


def get_new_housing_construction_data():
    """
    Extracts new housing construction data from the BFS Excel file.
    
    Returns:
        pd.DataFrame: DataFrame containing year and number of new constructions
    """
    import requests
    from io import BytesIO
    import pandas as pd
    
    url = r'https://dam-api.bfs.admin.ch/hub/api/dam/assets/32106363/master'
    
    response = requests.get(url)
    
    # Convert response content to bytes buffer
    content = BytesIO(response.content)
    
    # Read all sheets from excel file
    excel_file = pd.ExcelFile(content)
    sheet_names = excel_file.sheet_names
    
    # Create a list to store data from each sheet
    data_list = []
    
    # Process each sheet
    for sheet_name in sheet_names:
        # Read the specific sheet
        df = pd.read_excel(content, sheet_name=sheet_name)
        
        # For sheets with index < 18, extract cell D12 (row 10, column 3)
        # For sheets with index >= 18, extract cell B12 (row 10, column 1)
        if sheet_names.index(sheet_name) < 18:
            # Check if the sheet has at least 12 rows and 4 columns
            if df.shape[0] >= 12 and df.shape[1] >= 4:
                # Extract value from cell D12 (row 11, column 3 in zero-based indexing)
                cell_value = df.iloc[10, 3]
                
                # Add to data list with sheet name
                data_list.append({
                    'sheet_name': sheet_name,
                    'cell_value': cell_value
                })
            else:
                print(f"Sheet '{sheet_name}' doesn't have cell D12")
        else:
            # Check if the sheet has at least 12 rows and 2 columns
            if df.shape[0] >= 12 and df.shape[1] >= 2:
                # Extract value from cell B12 (row 11, column 1 in zero-based indexing)
                cell_value = df.iloc[10, 1]
                
                # Add to data list with sheet name
                data_list.append({
                    'sheet_name': sheet_name,
                    'cell_value': cell_value
                })
            else:
                print(f"Sheet '{sheet_name}' doesn't have cell B12")
    
    # Create a dataframe from the collected data
    result_df = pd.DataFrame(data_list)
    
    # Extract the first 4 characters from sheet_name and convert to datetime
    result_df['date'] = result_df['sheet_name'].str[:4].astype(int)
    result_df['date'] = pd.to_datetime(result_df['date'].astype(str)+'-01-01', format='%Y-%m-%d', errors='coerce')
    
    # Convert cell_value to numeric
    result_df['cell_value'] = pd.to_numeric(result_df['cell_value'], errors='coerce')
    result_df= result_df.rename(columns={'cell_value': 'new_constructions'})
    
    # Sort by year
    result_df = result_df.sort_values('date')

    # Keep only date and new_constructions collumns
    result_df = result_df.drop(columns={'sheet_name'})
    
    # Group by date and take the mean of duplicate dates
    result_df = result_df.groupby('date', as_index=False).mean()
    
    return result_df

def get_urban_population_data():
    """
    Downloads and processes urban population data from the Swiss Federal Statistical Office.
    
    Returns:
        pandas.DataFrame: A dataframe containing urban population data by year.
    """
    # Download the Excel file
    response = requests.get(url=r'https://dam-api.bfs.admin.ch/hub/api/dam/assets/32229112/master')
    excel_data = io.BytesIO(response.content)
    
    # Read the excel file
    df  =pd.read_excel(excel_data)

    dates = df.iloc[2, 1:].values  # Row 4 starts at index 3, skip first column

    # Extract values from row 6
    values = df.iloc[4, 1:].values  # Row 6 starts at index 5, skip first column

    # Create a new dataframe with the data in long format
    urban_population_df = pd.DataFrame({
        'date': dates,
        'pop_in_agglomeration': values
    })

    # Clean up the dataframe
    urban_population_df = urban_population_df.dropna()

    # Extract first 4 characters of date and convert to year
    urban_population_df['date'] = urban_population_df['date'].astype(str).str[:4].astype(int)
    urban_population_df['date'] = pd.to_datetime(urban_population_df['date'].astype(str)+'-01-01', format='%Y-%m-%d', errors='coerce')

    # Drop the last row
    urban_population_df = urban_population_df.iloc[:-1]
    
    return urban_population_df

def get_divorce_data():
    """
    Downloads and processes divorce data from the Swiss Federal Statistical Office.
    
    Returns:
        pandas.DataFrame: A dataframe containing divorce statistics by year.
    """

    # Get the API response
    response = requests.get(url=r"https://www.pxweb.bfs.admin.ch/api/v1/fr/px-x-0102020203_110/px-x-0102020203_110.px")

    # Parse the JSON data
    data = json.loads(response.text)

    # Extract the data into a DataFrame
    if 'variables' in data:
        # Process the metadata and structure
        variables = data['variables']
        
        # Extract years and indicators
        years = variables[0]['valueTexts']
        indicators = variables[1]['valueTexts']
        
        # Create a query to get the actual data
        query = {
            "query": [
                {
                    "code": "Jahr",
                    "selection": {
                        "filter": "item",
                        "values": variables[0]['values']
                    }
                },
                {
                    "code": "Demografisches Merkmal und Indikator",
                    "selection": {
                        "filter": "item",
                        "values": variables[1]['values']
                    }
                }
            ],
            "response": {
                "format": "json-stat2"
            }
        }
        
        # Make a POST request to get the actual data
        data_response = requests.post(
            url=r"https://www.pxweb.bfs.admin.ch/api/v1/fr/px-x-0102020203_110/px-x-0102020203_110.px",
            json=query
        )
        
        # Parse the response data
        data_json = json.loads(data_response.text)
        
        # Create a dataframe
        rows = []
        if 'value' in data_json:
            values = data_json['value']
            size0 = len(variables[0]['values'])
            size1 = len(variables[1]['values'])
            
            for i in range(size0):
                for j in range(size1):
                    index = i * size1 + j
                    if index < len(values):
                        rows.append({
                            'date': years[i],
                            'indicator': indicators[j],
                            'value': values[index]
                        })
        
        divorces_df = pd.DataFrame(rows)
        
        # Convert year to datetime and value to numeric
        divorces_df['date'] = pd.to_datetime(divorces_df['date'], format='%Y')
        divorces_df['value'] = pd.to_numeric(divorces_df['value'], errors='coerce')
        
        # Pivot the dataframe to have indicators as columns
        divorces_df = divorces_df.pivot(index='date', columns='indicator', values='value').reset_index()
        
        # Keep only the year and Divorces - Total columns
        divorces_df = divorces_df[['date', 'Divorces - Total']]
    else:
        print("JSON structure is different than expected. Examine the data:")
        print(data.keys())
        divorces_df = pd.DataFrame()
    
    return divorces_df

########################################################################################
# Getting and merging all data
########################################################################################


def get_and_merge_all_data():
    """
    Fetches all data from different sources and merges them on the date column.
    Returns a single dataframe with all variables joined.
    """
    # Get Housing Data
    dfs = get_clean_housing_data()
    housing_df = dfs['housing_df']
    
    # Get Demographics Data
    raw_demographics_df = get_demographics_data()
    demographics_df = long_to_wide_demographics(raw_demographics_df)
    
    # Get Interest Rates Data
    monthly_interest_rates_df = get_and_process_snb_data(url="https://data.snb.ch/api/cube/zimoma/data/json/en")
    interest_rates_df = group_data_by_year(monthly_interest_rates_df)
    
    # Get Monetary Base Data
    monthly_monetary_base_df = get_and_process_snb_data(url="https://data.snb.ch/api/cube/snbmoba/data/json/en")
    monetary_base_df = group_data_by_year(monthly_monetary_base_df)
    
    # Get Monetary Aggregates Data
    monthly_monetary_aggregate_df = get_and_process_snb_data_two_header_levels(url="https://data.snb.ch/api/cube/snbmonagg/data/json/en")
    monetary_aggregate_df = group_data_by_year(monthly_monetary_aggregate_df)
    
    # Get Mortgage Data
    monthly_mortage_df = get_and_process_mortgage_data(url="https://data.snb.ch/api/cube/bakredinausbm/data/json/en")
    mortgage_df = group_data_by_year(monthly_mortage_df)
    
    # Get GDP Data
    q_gdp_df = get_and_process_snb_data_two_header_levels(url="https://data.snb.ch/api/cube/gdpgnp/data/json/en")
    q_gdp_df = q_gdp_df[['date', 'Value, in CHF millions - Gross domestic product']]
    gdp_df = group_data_by_year_sum(q_gdp_df)
    this_year = datetime.datetime.now().year
    gdp_df = gdp_df[gdp_df['date'].dt.year != this_year]
    
    # Get Population Data
    population_df = get_swiss_population_data()
    
    # Get inflation Data
    monthly_inflation_df = get_and_process_snb_data(url="https://data.snb.ch/api/cube/plkoprinfla/data/json/en")
    monthly_inflation_df = monthly_inflation_df[['date', 'SFSO - Inflation according to the national consumer price index']]
    inflation_df = group_data_by_year(monthly_inflation_df)
    
    # Merge all dataframes on date
    merged_df = housing_df.copy()
    dataframes = [
        demographics_df,
        interest_rates_df, 
        monetary_base_df,
        monetary_aggregate_df,
        mortgage_df,
        gdp_df,
        population_df,
        inflation_df
    ]
    
    for df in dataframes:
        merged_df = merged_df.merge(df, on='date', how='outer')
    
    
    
    return merged_df

########################################################################################
# Initial Filtering of data (based on domain knowledge)
########################################################################################

def filter_df(merged_data_df):
    """
    Filters the merged dataframe based on domain knowledge.
    """
    # This is when asking_price begins. Want to predict 2025
    filtered_df = merged_data_df[(merged_data_df['date'] > '1970-01-01') & (merged_data_df['date'] < '2025-01-01')]
    
    # Removing some uneccesary collumm based on intuition and theory
        # - Let's focus on Houses and Appartments for prediction -> removing appartment buildings
        # - Changes because I already have levels
        # - Disagregated morgage data because I will include the total
        # - Seasonal and statistical adjustements because they may add noise
        # - Non CH interest rates
        # - Net measures of immi/emigtation and births/death because I will include Natural change and net migration
        # - Short term interest rates: keeping Switzerland - CHF - Call money rate (Tomorrow next) - 1 day because more data availability
        # - 'Level - Savings deposits', 'Level - Sight deposits',  'Level - Time deposits' because they are parts of a broader measure that I'm keeping
        # - Different ways of quantifiying loans because I'm keeping Mortages and Total Loans which seem more pertinent and broad.
        # - M1 and M2 Money Supply because they are narrower measures than M3
    lean_merged_data_df = filtered_df.drop(columns=[
        'Apartment buildings (residential investment property) - Transaction price',
        'Change from the corresponding month of the previous year - Currency in circulation',
        'Change from the corresponding month of the previous year - Deposits in transaction accounts',
        'Change from the corresponding month of the previous year - Monetary aggregate M1',
        'Change from the corresponding month of the previous year - Monetary aggregate M2',
        'Change from the corresponding month of the previous year - Monetary aggregate M3',
        'Change from the corresponding month of the previous year - Savings deposits',
        'Change from the corresponding month of the previous year - Sight deposits',
        'Change from the corresponding month of the previous year - Time deposits',
        'Banks in Switzerland - Domestic - Total loans - Utilisation',
        'Banks in Switzerland - Domestic - Total loans - Credit lines',
        'Banks in Switzerland - Domestic - Mortgage loans - Utilisation',
        'Banks in Switzerland - Domestic - Other loans - Total - Utilisation',
        'Banks in Switzerland - Domestic - Other loans - secured - Utilisation',
        'Banks in Switzerland - Domestic - Other loans - unsecured - Utilisation',
        'Banks in Switzerland - Foreign - Total loans - Utilisation',
        'Banks in Switzerland - Foreign - Total loans - Credit lines',
        'Banks in Switzerland - Foreign - Mortgage loans - Utilisation',
        'Banks in Switzerland - Foreign - Other loans - Total - Utilisation',
        'Banks in Switzerland - Foreign - Other loans - secured - Utilisation',
        'Banks in Switzerland - Foreign - Other loans - unsecured - Utilisation',
        'Statistical adjustments',
        'Seasonally adjusted - Seasonal factor',
        'United States - USD - SOFR - 1-day',
        'United States - USD - USD LIBOR - 3-month',
        'United Kingdom - GBP - GBP LIBOR - 3-month',
        'Euro area - EUR - ESTR - 1-day', 'Euro area - EUR - EURIBOR - 3-month',
        'Euro area - EUR - EUR LIBOR - 3-month',
        'Japan - JPY - TONA - 1-day',	
        'Japan - JPY - JPY LIBOR - 3-month',	
        'United Kingdom - GBP - SONIA - 1-day',
        'Deaths', 
        'Emigration',
        'Immigration', 
        'Live births',
        'Switzerland - CHF - SARON - 1 day',
        'Switzerland - CHF - Money market debt register claims of the Swiss Confederation - 3-month',
        'Switzerland - CHF - CHF LIBOR - 3-month',
        'Level - Savings deposits',
        'Level - Sight deposits', 
        'Level - Time deposits',
        'Banks in Switzerland - Total domestic and foreign - Total loans - Credit lines',
        'Banks in Switzerland - Total domestic and foreign - Other loans - Total - Utilisation',
        'Banks in Switzerland - Total domestic and foreign - Other loans - secured - Utilisation',
        'Banks in Switzerland - Total domestic and foreign - Other loans - unsecured - Utilisation',
        'Level - Monetary aggregate M1', 
        'Level - Monetary aggregate M2',
        'Utilisation - Monetary base', 
        'Seasonally adjusted - Monetary base',
        ])

    lean_merged_data_df = lean_merged_data_df.rename(columns={
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
    
    return lean_merged_data_df

