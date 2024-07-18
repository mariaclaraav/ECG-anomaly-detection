import os
from src.models.IsolationForest import IsoForestModel
from src.data.make_dataset import DataLoader

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
