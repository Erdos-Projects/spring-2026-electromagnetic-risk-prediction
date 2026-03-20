import pandas as pd
import numpy as np

def load_raw_nasa_omni_historical(year):
    r"""
    Load historical NASA OMNI data for a given year as a dataframe with columns
    ['Year', 'DOY', 'Hour', 'IMF_Mag', 'Bz_GSE', 'Proton_Density', 'Plasma_Speed', 'Kp_index'].

    Information about the dataset is available here: https://omniweb.gsfc.nasa.gov/html/ow_data.html
    """
    year = str(year)
    url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
    cols = ['Year', 'DOY', 'Hour', 'IMF_Mag', 'Bz_GSE', 'Proton_Density', 'Plasma_Speed', 'Kp_index']
    # Specific indices: 0,1,2 (Time), 12 (IMF), 15 (Bz), 23 (Density), 24 (Speed), 38 (Kp)
    df = pd.read_csv(url, sep='\s+', header=None, usecols=[0, 1, 2, 12, 15, 23, 24, 38], names=cols)
    return df

import pandas as pd
import requests
import os
import json
from datetime import datetime
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_and_log_omni_data(year, output_dir='data/raw/'):
    """
    Downloads raw OMNI data, saves it to disk, and returns dataframe if successful.

    Load historical NASA OMNI data for a given year as a dataframe with columns
    ['Year', 'DOY', 'Hour', 'IMF_Mag', 'Bx_GSE', 'By_GSE', 'Bz_GSE', 'Proton_Density', 'Plasma_Speed', 'Kp_index'].

    Information about the dataset is available here: https://omniweb.gsfc.nasa.gov/html/ow_data.html
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Provenance metadata
    url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
    timestamp = datetime.now().isoformat()
    raw_filename = f"omni2_{year}_raw.csv"
    raw_path = os.path.join(output_dir, raw_filename)
    
    logging.info(f"Starting download for year {year} from {url}")
    
    try:
        # 1. Acquisition
        cols = ['Year', 'DOY', 'Hour', 'IMF_Mag', 'Bx_GSE', 'By_GSE', 'Bz_GSE', 'Proton_Density', 'Plasma_Speed', 'Kp_index']
        # Specific indices from NASA documentation
        df = pd.read_csv(url, sep='\s+', header=None, usecols=[0, 1, 2, 12, 13, 14, 15, 23, 24, 38], names=cols)
        
        # 2. Save Raw Data (The "Documentation of Provenance" requirement)
        df.to_csv(raw_path, index=False)
        logging.info(f"Successfully saved raw data to {raw_path}")
        
        # 3. Create Metadata/Provenance Log
        metadata = {
            "year": year,
            "source_url": url,
            "download_timestamp": timestamp,
            "local_path": raw_path,
            "row_count": len(df),
            "license": "Public Domain (NASA SPDF)",
            "ethical_considerations": "Non-sensitive physical measurement data; no PII."
        }
        
        meta_path = os.path.join(output_dir, f"omni2_{year}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        return df

    except Exception as e:
        logging.error(f"Failed to fetch data for {year}: {e}")
        raise

def load_raw_omni_historical_if_exists(year, output_dir='data/raw/'):
    """
    Checks if the raw data file for the given year exists locally. If it does, loads it into a dataframe.
    If not, fetch from the NASA OMNI source and save it, then return the dataframe.
    """
    raw_filename = f"omni2_{year}_raw.csv"
    raw_path = os.path.join(output_dir, raw_filename)
    
    if os.path.exists(raw_path):
        logging.info(f"Found existing raw data file for year {year} at {raw_path}. Loading...")
        df = pd.read_csv(raw_path)
        return df
    else:
        return fetch_and_log_omni_data(year, output_dir)
    
def clean_nasa_omni_historical(df):
    r"""
    Clean the NASA OMNI historical dataframe by replacing missing values with NaN and converting data types.
    """
    df['Timestamp'] = pd.to_datetime(df['Year'], format='%Y') + \
                    pd.to_timedelta(df['DOY'] - 1, unit='D') + \
                    pd.to_timedelta(df['Hour'], unit='h')
    df.set_index('Timestamp', inplace=True)

    # Handle 'Ghost' Data (NASA fill values)
    # Kp is 0-90, Bx, By, Bz are ~ +/- 50, Speed is ~300-1000. 999s are errors.
    df = df.replace([99.9, 999.9, 9999., 99], np.nan)
    df = df.interpolate(method='linear')

    # Replace missing values (e.g., 999.99) with NaN
    df.replace(999.99, np.nan, inplace=True)
    
    # Kp index is given as 0-90, we convert it to a real Kp value by dividing by 10
    # df['Kp_real'] = df['Kp_index'] / 10.0    
    
    return df

def load_and_clean_nasa_omni_historical(year, output_dir='data/raw/'):
    """
    Load and clean NASA OMNI historical data for a given year. This function checks for existing raw data,
    fetches if necessary, and then applies cleaning steps to return a ready-to-use dataframe.
    """
    df_raw = load_raw_omni_historical_if_exists(year, output_dir)
    df_clean = clean_nasa_omni_historical(df_raw)
    return df_clean


def load_and_clean_nasa_omni_historical_for_years(years, output_dir='data/raw/'):
    """
    Load and clean NASA OMNI historical data for multiple years, returned as a list of dataframes.
    """
    yearly_dataframes = []
    for year in years:
        df_year = load_and_clean_nasa_omni_historical(year, output_dir)
        yearly_dataframes.append(df_year)

    return yearly_dataframes    

def merge_yearly_dataframes(yearly_dataframes, drop_duplicate_index=True, sort_index=True):
    """
    Merge multiple yearly dataframes into a single dataframe.
    """
    if not yearly_dataframes:
        raise ValueError("yearly_dataframes must contain at least one dataframe")

    merged_df = pd.concat(yearly_dataframes, axis=0)

    if drop_duplicate_index:
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]

    if sort_index:
        merged_df = merged_df.sort_index()

    return merged_df

