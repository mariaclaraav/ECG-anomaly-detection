import os
from src.models.IsolationForest import IsoForestModel
from src.models.LSTMAutoencoder import LSTMAutoencoder
from src.data.make_dataset import DataLoader, DataProcessingAE

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
    
    # Initialize data processor for Autoencoder and process it
    data_processor = DataProcessingAE(data_path)
    data_processor.load_data()
    data_processor.preprocess_data()
    X_train, y_train, X_val, y_val, X_test, y_test = data_processor.prepare_autoencoder_data()    
    train_data, val_data, test_data = data_processor.scale_data(X_train, X_val, X_test)
    
    # Define the number of timesteps and features for the LSTM data input shape
    timesteps = 10
    n_features = X_train.shape[1] // timesteps
    X_train = X_train.iloc[:, :n_features * timesteps]
    X_train = X_train.values.reshape((X_train.shape[0], timesteps, n_features))
    X_val = X_val.iloc[:, :n_features * timesteps]
    X_val = X_val.values.reshape((X_val.shape[0], timesteps, n_features))
    X_test = X_test.iloc[:, :n_features * timesteps]
    X_test = X_test.values.reshape((X_test.shape[0], timesteps, n_features))

    # Initialize and train LSTM autoencoder
    lstm_autoencoder = LSTMAutoencoder(timesteps, n_features, neurons_1=128, neurons_2=64, epochs=60, batch_size=32)
    history = lstm_autoencoder.train(X_train, X_val)
    
    # Plot the training and validation loss
    lstm_autoencoder.plot_loss(history)
    
    # Reconstruct validation data and plot results for samples
    for i in range(1, 5):
        lstm_autoencoder.reconstruct(X_val, i, N=X_val.shape[0], M=n_features * timesteps)

    # Define the threshold for anomaly detection based on validation data
    threshold = lstm_autoencoder.define_threshold(X_val, N=X_val.shape[0], M=n_features * timesteps)

    # Detect anomalies in test data and plot results for samples
    for i in range(1, 5):
        lstm_autoencoder.detect_anomalies(X_test, threshold, i, N=X_test.shape[0], M=n_features * timesteps)

    # Detect anomalies in validation data and plot results for samples
    for i in range(1, 5):
        lstm_autoencoder.detect_anomalies(X_val, threshold, i, N=X_val.shape[0], M=n_features * timesteps)    
