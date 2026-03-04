import requests
import pandas as pd
import os
import json
import logging
from datetime import datetime

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_noaa_realtime_plasma(output_dir='data/raw/realtime/'):
    """
    Fetches real-time solar wind data from NOAA SWPC ACE API.
    ACE is positioned at L1 and provides earlier warning than DSCOVR.

    Information about datasets:
    - MAG : https://services.swpc.noaa.gov/text/ace-magnetometer.txt
    - SWEPAM : https://services.swpc.noaa.gov/text/ace-swepam.txt
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ACE endpoints (MAG and SWEPAM)
    mag_url = "https://services.swpc.noaa.gov/json/ace/mag/ace_mag_1h.json"
    swepam_url = "https://services.swpc.noaa.gov/json/ace/swepam/ace_swepam_1h.json"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.info(f"Accessing NOAA SWPC ACE API")
    
    try:
        # 1. Fetch MAG and SWEPAM data
        mag_response = requests.get(mag_url, timeout=10)
        swepam_response = requests.get(swepam_url, timeout=10)
        mag_response.raise_for_status()
        swepam_response.raise_for_status()
        
        mag_json = mag_response.json()
        swepam_json = swepam_response.json()
        
        # 2. Save Raw Data with provenance
        raw_mag_path = os.path.join(output_dir, f"ace_mag_{timestamp}.json")
        raw_swepam_path = os.path.join(output_dir, f"ace_swepam_{timestamp}.json")
        
        with open(raw_mag_path, 'w') as f:
            json.dump(mag_json, f)
        with open(raw_swepam_path, 'w') as f:
            json.dump(swepam_json, f)
        logging.info(f"Raw MAG archived at {raw_mag_path}")
        logging.info(f"Raw SWEPAM archived at {raw_swepam_path}")
        
        # 3. Create Provenance Metadata
        metadata = {
            "source": "NOAA Space Weather Prediction Center (SWPC)",
            "satellite": "ACE (Advanced Composition Explorer)",
            "location": "L1 (~1.5M km from Earth)",
            "endpoints": [mag_url, swepam_url],
            "access_timestamp": datetime.now().isoformat(),
            "data_type": "JSON",
            "usage_notes": "MAG provides Bz measurements; SWEPAM provides speed and density. Merged on time_tag."
        }
        
        meta_path = os.path.join(output_dir, f"ace_metadata_{timestamp}.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        # 4. Merge datasets
        df_mag = pd.DataFrame(mag_json[1:], columns=mag_json[0])
        df_swepam = pd.DataFrame(swepam_json[1:], columns=swepam_json[0])
        df = pd.merge(df_mag, df_swepam, on='time_tag')
        
        logging.info(f"Acquired {len(df)} real-time ACE data points.")
        
        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"API Error: {e}")
        return None


