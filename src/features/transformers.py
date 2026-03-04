import pandas as pd
import numpy as np

class SolarWindTransformer:
    """
    Applies physics-based feature engineering to solar wind data.
    """
    
    @staticmethod
    def engineer_southward_bz(df, bz_column='gse_bz'):
        """
        Extract southward Bz component (negative Bz is geoeffective).
        Southward Bz = |min(Bz, 0)| = clip negative values and take absolute value.
        """
        df['Bz_South'] = df[bz_column].clip(upper=0).abs()
        return df
    @staticmethod
    def engineer_magnetic_strength(df, bx_column='gse_bx', by_column='gse_by', bz_column='gse_bz'):
        """
        Compute total magnetic field strength: B_total = sqrt(Bx^2 + By^2 + Bz^2)
        """
        df['B_total'] = (df[bx_column]**2 + df[by_column]**2 + df[bz_column]**2)**0.5
        return df

    @staticmethod
    def engineer_azimuthal_angle(df, by_column='gse_by', bz_column='gse_bz'):
        """
        Compute azimuthal angle of the magnetic field in the YZ plane: atan2(By, Bz)
        """
        df['B_azimuth'] = np.arctan2(df[by_column], df[bz_column])
        return df

    @staticmethod
    def engineer_energy_flux(df, speed_column='speed', bz_south_column='Bz_South'):
        """
        Compute solar wind energy flux (proxy for storm impact).
        Energy_Flux = Plasma_Speed * Bz_South
        """
        df['Energy_Flux'] = df[speed_column] * df[bz_south_column]
        return df
    
    @staticmethod
    def compute_l1_lag(df, speed_column='speed', distance_km=1500000):
        """
        Compute propagation lag from L1 to Earth.
        
        Args:
            df: DataFrame with solar wind speed
            speed_column: name of speed column (km/s)
            distance_km: L1 to Earth distance (default 1.5M km)
        
        Returns:
            DataFrame with lag_min column added
        """
        df['lag_min'] = (distance_km / df[speed_column]) / 60
        return df
    
    @staticmethod
    def apply_impact_shift(df, time_column='time', lag_column='lag_min'):
        """
        Shift observation time by L1 lag to get Earth impact time.
        """
        df['Impact_Time'] = df[time_column] + pd.to_timedelta(df[lag_column], unit='m')
        return df
    
    @staticmethod
    def compute_energy_flux_rolling_average(df, window_hours=3):
        """
        Compute rolling average of energy flux to capture sustained conditions.
        """
        df[f'Energy_{window_hours}h_Avg'] = df['Energy_Flux'].rolling(window=window_hours).mean()
        return df
    
    @staticmethod
    def compute_speed_rolling_max(df, window_hours=6):
        """
        Compute rolling maximum of plasma speed to capture peak conditions.
        """
        df[f'Speed_{window_hours}h_Max'] = df['Plasma_Speed'].rolling(window=window_hours).max()
        return df

    @staticmethod
    def transform(df, speed_col='Plasma_Speed', bx_col='Bx_GSE', by_col='By_GSE', bz_col='Bz_GSE', time_col='Timestamp'):
        """
        Apply all transformations in sequence.
        """
        df = SolarWindTransformer.compute_l1_lag(df, speed_column=speed_col)
        df = SolarWindTransformer.engineer_southward_bz(df, bz_column=bz_col)
        df = SolarWindTransformer.engineer_energy_flux(df, speed_column=speed_col, bz_south_column='Bz_South')
        df = SolarWindTransformer.compute_energy_flux_rolling_average(df, window_hours=3)
        df = SolarWindTransformer.compute_speed_rolling_max(df, window_hours=6)
        df = SolarWindTransformer.engineer_magnetic_strength(df, bx_column=bx_col, by_column=by_col, bz_column=bz_col)
        df = SolarWindTransformer.engineer_azimuthal_angle(df, by_column=by_col, bz_column=bz_col)
        return df