# spring-2026-electromagnetic-risk-prediction
Team project: spring-2026-electromagnetic-risk-prediction

## Overview
Geomagnetic storms driven by solar activity pose risks to power grids, satellites, and communication systems. In this project, we will develop a data-driven solar-to-ground proxy model that predicts near-term geomagnetic activity using continuous solar activity data. We would like to make robust predictions about space weather effects on earth while remaining computationally efficient and suitable for rapid deployment.

**Core Question:** Can we predict geomagnetic storms ($K_p \geq 5.0$) with sufficient lead time using L1 solar wind observations? See `kpis.md`.

## Data Inventory & Provenance
| Source | Access Method | Frequency | License |
| :--- | :--- | :--- | :--- |
| **NASA OMNIWeb** | HTTPS/CSV (`src/data/fetch_nasa_omni_historical.py`) | Hourly (Historical) | Public Domain |
| **NOAA SWPC** | JSON API (Real-time stream) (`src/data/fetch_noaa_realtime.py`) | 1-Minute | Public Domain |


