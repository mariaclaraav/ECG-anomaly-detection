import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        df_normal = pd.read_parquet(self.data_path + 'ptdb_normal.parquet')
        df_anomaly = pd.read_parquet(self.data_path + 'ptdb_abnormal.parquet')
        df_normal['label'] = 0
        df_anomaly['label'] = 1
        return df_normal, df_anomaly

    def preprocess_data(self, df_normal, df_anomaly):  
        data = pd.concat([df_normal, df_anomaly], axis=0).reset_index(drop=True)
        
        labels = data['label'].values
        data = data.drop('label', axis=1)
        
        data = data.replace(0, np.nan)
        data = data.fillna(method='ffill').fillna(method='bfill')
        data = data.dropna(axis=1, how='all')
        
        if data.empty:
            raise ValueError("Dataframe is empty after handling NaN values")   
        
        return data, labels