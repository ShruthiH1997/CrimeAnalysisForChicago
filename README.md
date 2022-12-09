# CrimeAnalysisForChicago
Crime Analysis and Prediction for Chicago Neighborhoods

# INVESTIGATING CRIMES IN CHICAGO NEIGHBORHOODS

## File Description 

### Dataset Files
Crime_merged.csv: The crime_merged data was created by merging Chicago Crime data with Community Snapshots data. Both the datasets can be found [here](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2) and [here](https://www.cmap.illinois.gov/data/community-snapshots)

del.csv: The del.csv file has the community areas data

### month_clusters.py
This file can be used to perform month-wise clustering on the data. 

### timeseries.py
Time series predictions using FB’s Prophet forecasting model. 


### Time_series_analysis_EMA_WMA.ipynb
This file contains a time series analysis of Chicago's weekly crime rate using Weighted Moving Average and Exponential Moving Average Time Series Analysis. We analyzed the four most prevalent categories of crime in Chicago and the top four community areas with the highest crime rate.

### requirements.txt
Contains all the required dependencies

## Streamlit Demo:
```
pip install -r requirements.txt
```

### Clustering:
```
streamlit run month_clusters.py
```

### Time series:
```
streamlit run timeseries.py
```


