import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import IsolationForest

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

class IsoForestModel:
    def __init__(self, contamination):
        print("\nInstantiating Isolation Forest model\n")        
        self.contamination = contamination
        self.model = None
    
    def scale_data(self, X_train, X_test):
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train_model(self, X_train):
        self.model = IsolationForest(contamination=self.contamination, random_state=42)
        self.model.fit(X_train)

    def predict_anomalies(self, X_train, X_test):
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        y_pred_train = np.where(y_pred_train == -1, 1, 0)
        y_pred_test = np.where(y_pred_test == -1, 1, 0)
        
        return y_pred_train, y_pred_test

    def plot_precision_recall_curve(self, y_test, y_pred_test):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_test)
        auprc_test = auc(recall, precision)
        print(f'AUPRC on the test set: {auprc_test:.3f}')
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'Test Set (AUPRC = {auprc_test:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_confusion_matrix(self, y_test, y_pred_test):
        plt.rcParams['font.family'] = 'Times New Roman'
        cm = confusion_matrix(y_test, y_pred_test, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
        disp.plot()
        plt.grid(True)
        plt.title('Confusion Matrix')
        plt.show()

    def plot_anomalies(self, X_test, y_test, y_pred_test):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.figure(figsize=(14, 8))
        
        # Plot predicted anomalies
        plt.subplot(2, 1, 1)
        plt.plot(X_test, label='Test Data')
        predicted_anomalies = np.where(y_pred_test == 1)
        plt.scatter(predicted_anomalies, X_test[predicted_anomalies], color='red', label='Predicted Anomalies', marker='o')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Test Data Predicted Anomalies')
        plt.grid(True)
        plt.legend()

        # Plot real anomalies
        plt.subplot(2, 1, 2)
        plt.plot(X_test, label='Test Data')
        real_anomalies = np.where(y_test == 1)
        plt.scatter(real_anomalies, X_test[real_anomalies], color='orange', label='Real Anomalies', marker='o')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Test Data with Real Anomalies')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
        
    def run(self, data, labels):
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        X_train_scaled, X_test_scaled = self.scale_data(X_train, X_test)
        
        self.train_model(X_train_scaled)
        y_pred_train, y_pred_test = self.predict_anomalies(X_train_scaled, X_test_scaled)
        
        self.plot_precision_recall_curve(y_test, y_pred_test)
        self.plot_confusion_matrix(y_test, y_pred_test)
        self.plot_anomalies(X_test_scaled[:, 188], y_test, y_pred_test)
        print('\nFinished\n')  
        print('-'*40)
        
if __name__ == "__main__":
    os.system('cls')
    
    data_path = 'C:/Users/maria/OneDrive/Documentos/Github/ECG-anomaly-detection/data/processed/'
    contamination = 0.5  # Set the contamination rate 
    
    # Load and preprocess data
    data_loader = DataLoader(data_path)
    df_normal, df_anomaly = data_loader.load_data()
    data, labels = data_loader.preprocess_data(df_normal, df_anomaly)
    
    # Run Isolation Forest model
    iso_forest_model = IsoForestModel(contamination)
    iso_forest_model.run(data, labels)
