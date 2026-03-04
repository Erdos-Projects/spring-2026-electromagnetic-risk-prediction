import pandas as pd
from src.features.transformers import SolarWindTransformer

def align_ace_to_omni_specs(df):
    """
    Aligns ACE MAG and SWEPAM data to match OMNI baseline specifications.
    """
    # 1. Type Casting (NOAA sends everything as strings)
    df['time'] = pd.to_datetime(df['time_tag'])
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df['density'] = pd.to_numeric(df['dens'], errors='coerce')
    df['gse_bz'] = pd.to_numeric(df['gse_bz'], errors='coerce')
    df['gse_by'] = pd.to_numeric(df['gse_by'], errors='coerce')
    df['gse_bx'] = pd.to_numeric(df['gse_bx'], errors='coerce')

    # 3. Resample to Hourly to ensure stability
    df_aligned = df.set_index('Impact_Time').resample('1h').mean(numeric_only=True)

    # 4. Rename columns to match baseline
    mapping = {
        'speed': 'Plasma_Speed',
        'density': 'Proton_Density',
        'gse_bz': 'Bz_GSE',
        'gse_by': 'By_GSE',
        'gse_bx': 'Bx_GSE'
    }
    df_aligned = df_aligned.rename(columns=mapping)

    return df_aligned[['Plasma_Speed', 'Proton_Density', 'Bx_GSE', 'By_GSE', 'Bz_GSE', 'Energy_Flux']]

def clean_inference_data(df, limit=3):
    """
    Strategy: 
    1. Linear interpolate small gaps (up to 'limit' hours).
    2. For large gaps, we MUST drop or flag, as the model cannot 'guess' solar wind.
    """
    # Interpolate small gaps (linear is standard for SW data)
    df_clean = df.interpolate(method='linear', limit=limit)
    
    # After interpolation, if rows are still NaN, it means the gap was too big.
    # We drop these because predicting a storm without a driver is high-risk.
    df_clean = df_clean.dropna()
    
    print(f"Cleaned Data: Dropped {len(df) - len(df_clean)} rows due to large gaps.")
    return df_clean