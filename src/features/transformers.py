import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class Event:
    start: pd.Timestamp
    end: pd.Timestamp  # inclusive

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
    def compute_newell_coupling(df):
        """
        Compute Newell coupling function: V^(4/3) * B^(2/3) * sin^(8/3)(theta/2)
        This represents the efficiency of solar wind energy coupling to the magnetosphere.
        """
        df['Newell_Coupling'] = (
            df['Plasma_Speed']**(4/3) * 
            df['B_total']**(2/3) * 
            np.abs(np.sin(df['B_azimuth'] / 2))**(8/3)
        )
        return df

    @staticmethod
    def compute_newell_integral(df, window=3):
        df[f'Newell_{window}H_Integral'] = df['Newell_Coupling'].rolling(window=window).sum()
        return df

    @staticmethod
    def compute_rolling_min_and_max(df, bz_col, window=3):
        df[f'Bz_{window}H_Max'] = df[bz_col].rolling(window=window).max()
        df[f'Bz_{window}H_Min'] = df[bz_col].rolling(window=window).min()
        df[f'B_std_{window}H'] = df['B_total'].rolling(window=window).std()
        return df

    @staticmethod
    def engineer_confidence_gates(df, threshold=40000):
        """
        Creates a 'sustained energy' feature. 
        Higher values = much higher Precision for storm events.
        """
        if threshold is None:
            threshold = 10000 # df['Newell_Coupling'].quantile(0.9)
        
        is_above = (df['Newell_Coupling'] > threshold).astype(int)
        df['Storm_Potential_High'] = is_above
        return df 

    @staticmethod
    def transform(df, speed_col='Plasma_Speed', bx_col='Bx_GSE', by_col='By_GSE', bz_col='Bz_GSE', time_col='Timestamp'):
        """
        Apply all transformations in sequence.
        """
        # df = SolarWindTransformer.compute_l1_lag(df, speed_column=speed_col)
        df = SolarWindTransformer.engineer_southward_bz(df, bz_column=bz_col)
        df = SolarWindTransformer.engineer_energy_flux(df, speed_column=speed_col, bz_south_column='Bz_South')
        df = SolarWindTransformer.compute_energy_flux_rolling_average(df, window_hours=3)
        df = SolarWindTransformer.compute_speed_rolling_max(df, window_hours=6)
        df = SolarWindTransformer.engineer_magnetic_strength(df, bx_column=bx_col, by_column=by_col, bz_column=bz_col)
        df = SolarWindTransformer.engineer_azimuthal_angle(df, by_column=by_col, bz_column=bz_col)
        df = SolarWindTransformer.compute_newell_coupling(df)
        df = SolarWindTransformer.compute_newell_integral(df, window=3)
        df = SolarWindTransformer.compute_rolling_min_and_max(df, bz_col=bz_col, window=3)
        df = SolarWindTransformer.engineer_confidence_gates(df, threshold=40000)
        return df


class StormEventExtractor:
    """
    Extracts storm events from Kp index time series.
    """
    STORM_KP_INDEX_THRESHOLD = 50

    @staticmethod
    def storm_bool_from_kp_index(kp_index): 
        kp_index = kp_index.astype(float)
        return kp_index >= STORM_KP_INDEX_THRESHOLD

    @staticmethod
    def extract_true_storm_events(storm):
        """
        Extract maximal contiguous True intervals.
        """
        if not isinstance(storm.index, pd.DatetimeIndex):
            raise ValueError("storm must have a DatetimeIndex")
        storm = storm.astype(bool).sort_index()

        starts = storm & ~storm.shift(1, fill_value=False)
        ends = storm & ~storm.shift(-1, fill_value=False)

        start_times = storm.index[starts].to_list()
        end_times = storm.index[ends].to_list()
        return [Event(s, e) for s, e in zip(start_times, end_times)]
