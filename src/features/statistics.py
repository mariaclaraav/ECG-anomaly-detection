import pandas as pd
from scipy.stats import kurtosis, skew
import numpy as np
import statsmodels.tsa.stattools as adf

def apply_adf(series):
    # Remove zero values added for padding from the series
    series = series[series != 0]
    
    adf_test = adf.adfuller(series)
    result = {
        'ADF Statistic': adf_test[0],
        'p-value': adf_test[1],
        'Critical Values': adf_test[4],
        'Is Stationary': adf_test[1] < 0.05  # Typically, a p-value < 0.05 indicates stationarity
    }
    return result

def test_adf_df(df, n=5):
    # Select n random rows/samples in datasetclasse
    random_samples = df.sample(n=n, random_state=40)
    results_adf = random_samples.apply(apply_adf, axis=1)
    return results_adf

def calculate_statistics(series):
    # Remove zero values from the series
    series = series[series != 0]
    mean = series.mean()
    std = series.std()
    kurt = kurtosis(series)
    skewness = skew(series)
    return mean, std, kurt, skewness

def calculate_stats(row):
    filtered_row = row[row != 0]  # Filter out the zeros
    if not filtered_row.empty:   # Check if the row still has elements
        mean = filtered_row.mean()
        std = filtered_row.std()
        kurt = kurtosis(filtered_row, fisher=True)  # By default, Fisher’s definition is used
        skewness = skew(filtered_row)
        return mean, std, kurt, skewness
    else:
        return None, None, None, None  # Return None for mean, std, kurtosis, and skewness if all values are zero