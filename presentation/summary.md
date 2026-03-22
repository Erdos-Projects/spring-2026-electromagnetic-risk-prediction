# Solar Storm Prediction Executive Summary
The Sun emits solar wind at the Earth sometimes resulting in geomagnetic storms. 
These storms can negatively affect power grids, satellites, and communication systems. 
Because of this, it is useful for operators to receive "early warning" about potential solar storms.
This project aims to predict solar storms from incoming solar wind data from NASA satellites located between the Earth and the Sun.
For this project, we ingested the satellite data with some robustness checks to handle inherent unreliability, engineered "physics aware" features, engaged in feature selection to drop statistically redundant and unuseful features, and ultimately trained an XGBoost classifier to predict storms at an acceptable level for weather prediction.
Ultimately, our model could be deployed to make prediction on realtime solar wind data from NOAA to predict solar storms with approximately 45 minute lead times.