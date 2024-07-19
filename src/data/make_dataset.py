import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
    
class DataProcessingAE:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df_normal = None
        self.df_anomaly = None
        self.scaler = MinMaxScaler()
    
    def load_data(self):
        self.df_normal = pd.read_parquet(self.data_path + 'ptdb_normal.parquet')
        self.df_anomaly = pd.read_parquet(self.data_path + 'ptdb_abnormal.parquet')
    
    def preprocess_data(self):
        self.df_normal = self.df_normal.replace(0, np.nan)
        self.df_anomaly = self.df_anomaly.replace(0, np.nan)

        self.df_normal.drop(columns=['Mean', 'Skewness', 'Kurtosis', 'Std'], inplace=True)
        self.df_anomaly.drop(columns=['Mean', 'Skewness', 'Kurtosis', 'Std'], inplace=True)

        self.df_normal = self.df_normal.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
        self.df_anomaly = self.df_anomaly.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

        self.df_normal = self.df_normal.dropna(axis=1, how='all')
        self.df_anomaly = self.df_anomaly.dropna(axis=1, how='all')

        self.df_normal['label'] = 0
        self.df_anomaly['label'] = 1

    def prepare_autoencoder_data(self):
        X_train = self.df_normal.iloc[:-809, :-1]
        y_train = self.df_normal.iloc[:-809]['label']

        X_val = self.df_normal.iloc[-809:, :-1]
        y_val = self.df_normal.iloc[-809:]['label']

        X_test = self.df_anomaly.iloc[:, :-1]
        y_test = self.df_anomaly['label']

        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        X_val.reset_index(drop=True, inplace=True)
        y_val.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        return X_train, y_train, X_val, y_val, X_test, y_test

    def scale_data(self, X_train, X_val, X_test):
        train_data = self.scaler.fit_transform(X_train)
        val_data = self.scaler.transform(X_val)
        test_data = self.scaler.transform(X_test)
        return train_data, val_data, test_data
