# spring-2026-electromagnetic-risk-prediction
Team project: spring-2026-electromagnetic-risk-prediction

## Overview
Geomagnetic storms driven by solar activity pose risks to power grids, satellites, and communication systems. In this project, we developed a data-driven solar-to-ground proxy model that predicts near-term geomagnetic activity using solar wind data. The final model is an XGBoost Classifier tuned to maximize correctly predicting when a storm occurs subject to keeping the false positive rate at an acceptable level. 

**Core Question:** Can we predict geomagnetic storms ($K_p \geq 5.0$) with using solar wind observations from NASA and NOAA? See `kpis.md` for details.

See `src/notebooks/final_results.ipynb` for final model training and evaluation.

## Data Inventory & Provenance
| Source | Access Method | Frequency | License |
| :--- | :--- | :--- | :--- |
| **NASA OMNIWeb** | HTTPS/CSV (`src/data/fetch_nasa_omni_historical.py`) | Hourly (Historical) | Public Domain |
| **NOAA SWPC** | JSON API (Real-time stream) (`src/data/fetch_noaa_realtime.py`) | 1-Minute | Public Domain |