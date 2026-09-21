# circulation-extremes

This repository  contains code for the manuscript titled "Linking Large-Scale Atmospheric Circulation Regimes and Compound Weather Extremes across Western North America" by Laverty et al. (2026). The associated datasets which this code uses are stored in the following open-access repository: https:XXX. 

Data comes from daily ERA5 reanalysis (Copernicus Climate Data Store): 500 hPa geopotential height, 2m maximum temperature, and total precipitation, 1940–2024, 20-80°N, 170-100°W.

The code stored here is organized into discrete sections of processing and analysis, as follows:
- Processing of Geopotential Height (Z500) data: detrending, climatology, anomalies, k-means clustering, regime persistence
- Processing of Temperature data: detrending, climatology, anomalies, standardized anomalies
- Processing of Precipitation data: climatology, anomalies, gamma fitting of wet and dry thresholds 
- Identifying Compound Extremes: climatological frequencies, regime anomalies and significance, changes between two periods (1941-1970, 1991-2020)
- Figures and Analysis: main-text and supplementary figures, trend analysis

Please feel free to contact me with any questions at amanda.laverty@wsu.edu
