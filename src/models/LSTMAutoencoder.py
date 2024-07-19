import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input
import matplotlib.pyplot as plt

class LSTMAutoencoder:
    def __init__(self, timesteps, n_features, neurons_1=128, neurons_2=64, epochs=60, batch_size=32):
        print("\nInstantiating the LSTM AE model\n") 
        self.timesteps = timesteps
        self.n_features = n_features
        self.neurons_1 = neurons_1
        self.neurons_2 = neurons_2
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Input(shape=(self.timesteps, self.n_features)),
            LSTM(self.neurons_1, activation='relu', return_sequences=True),
            LSTM(self.neurons_2, activation='relu', return_sequences=False),
            RepeatVector(self.timesteps),
            LSTM(self.neurons_2, activation='relu', return_sequences=True),
            LSTM(self.neurons_1, activation='relu', return_sequences=True),
            TimeDistributed(Dense(self.n_features))
        ])
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        return model

    def train(self, train_data, val_data):
        history = self.model.fit(
            train_data, train_data, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            validation_data=(val_data, val_data),
            shuffle=True
        )
        return history

    def plot_loss(self, history):
        plt.figure(figsize=(12, 4))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

    def reconstruct(self, test_data, sample_index, N, M):
        decoded_data = self.model.predict(test_data)
        decoded_data = decoded_data.reshape(N, M)
        test_data = test_data.reshape(N, M)
        t = np.arange(0, M, 1)
        
        plt.figure(figsize=(12, 4))
        plt.plot(t, test_data[sample_index], 'b', label="Input")
        plt.plot(t, decoded_data[sample_index], 'r', label="Reconstruction")
        plt.fill_between(np.arange(len(test_data[sample_index])), decoded_data[sample_index], test_data[sample_index], color='lightcoral', label="Error")
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('LSTM AE', size=14)
        plt.legend()
        plt.grid(True)
        plt.show()

    def define_threshold(self, val_data, N, M):
        val_predictions = self.model.predict(val_data)
        val_predictions = val_predictions.reshape(N, M)
        val_data = val_data.reshape(N, M)
        
        val_errors = np.square(val_data - val_predictions).mean(axis=1)
        error = np.square(val_data - val_predictions).flatten()
        
        threshold = val_errors.mean() + 8 * val_errors.std()
        
        plt.figure(figsize=(8, 4))
        plt.hist(error, bins=200, alpha=0.75, color='blue', edgecolor='black', density=True)
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Validation Error Distribution')
        plt.xlim([None, 0.10])
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return threshold

    def detect_anomalies(self, test_data, threshold, sample_index, N, M):
        decoded_data = self.model.predict(test_data)
        decoded_data = decoded_data.reshape(N, M)
        test_data = test_data.reshape(N, M)
        t = np.arange(0, M, 1)
        
        val_mse = np.square(test_data - decoded_data)
        anomalies = val_mse > threshold
        
        if np.any(anomalies[sample_index]):
            print("Anomalies detected in the series.")
        else:
            print("No anomalies detected in the series.")
        
        plt.figure(figsize=(12, 4))
        plt.plot(t, test_data[sample_index], 'b', label="Input")
        plt.plot(t, decoded_data[sample_index], 'r', label="Reconstruction")
        
        anomaly_indices = np.where(anomalies[sample_index])[0]
        if anomaly_indices.size > 0:
            plt.scatter(anomaly_indices, test_data[sample_index][anomaly_indices], color='orange', s=40, label='Anomaly')
        
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('LSTM AE with Anomalies', size=14)
        plt.legend()
        plt.grid(True)
        plt.show()
