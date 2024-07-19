# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input

from sklearn.model_selection import train_test_split
plt.rcParams['font.family'] = 'Times New Roman'

# %%
data_path = 'C:/Users/maria/OneDrive/Documentos/Github/ECG-anomaly-detection/data/processed/'
df_normal = pd.read_parquet(data_path + 'ptdb_normal.parquet')
df_anomaly = pd.read_parquet(data_path + 'ptdb_abnormal.parquet')

# %%
df_normal = df_normal.replace(0, np.nan)
df_anomaly = df_anomaly.replace(0, np.nan)

# %%
df_normal.drop(columns=['Mean','Skewness','Kurtosis','Std'],inplace=True)
df_anomaly.drop(columns=['Mean','Skewness','Kurtosis','Std'], inplace=True)

# %%
df_normal = df_normal.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
df_anomaly = df_anomaly.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

# %%
# Drop columns with all NaN values
df_normal = df_normal.dropna(axis=1, how='all')
df_anomaly = df_anomaly.dropna(axis=1, how='all')

# %%
df_normal['label'] = 0
df_anomaly['label'] = 1

# %%
def prepare_autoencoder_data(normal_df, anomaly_df):
       
    X_train = normal_df.iloc[:-809,:-1]
    y_train = normal_df.iloc[:-809]['label']
    
    # Separating the features and labels of normal data for validation
    
    X_val = normal_df.iloc[-809:, :-1]
    y_val = normal_df.iloc[-809:]['label']
    
    # Separating the features and labels of anomaly data
    X_test = anomaly_df.iloc[:, :-1]
    y_test = anomaly_df['label']   
        
    # Resetting the indices of the DataFrames
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = prepare_autoencoder_data(df_normal, df_anomaly)

print(f'Training set size: {X_train.shape}')
print(f'Validation set size: {X_val.shape}')
print(f'Test set size: {X_test.shape}')

scaler = MinMaxScaler()
train_data = scaler.fit_transform(X_train)
val_data = scaler.transform(X_val)
test_data = scaler.transform(X_test)

# %% [markdown]
# ### LSTM input 
# 
# Here we transform the input data into the format expected by LSTM models, which work with sequences of timesteps.

# %%
df_normal

# %%
timesteps = 10  # Adjust this value to see its effect
n_features = X_train.shape[1] // timesteps

X_train = X_train.iloc[:, :n_features * timesteps]
X_train = X_train.values.reshape((X_train.shape[0], timesteps, n_features))

X_val = X_val.iloc[:, :n_features * timesteps]
X_val = X_val.values.reshape((X_val.shape[0], timesteps, n_features))

X_test = X_test.iloc[:, :n_features * timesteps]
X_test = X_test.values.reshape((X_test.shape[0], timesteps, n_features))

# %%
train_data = tf.cast(X_train, tf.float32)
val_data = tf.cast(X_val, tf.float32)
test_data = tf.cast(X_test, tf.float32)

# %%
# Build LSTM Autoencoder
lstm_autoencoder = Sequential([
    Input(shape=(timesteps, n_features)),
    LSTM(128, activation='relu', return_sequences=True),
    LSTM(64, activation='relu', return_sequences=False),
    RepeatVector(timesteps),
    LSTM(64, activation='relu', return_sequences=True),
    LSTM(128, activation='relu', return_sequences=True),
    TimeDistributed(Dense(n_features))
])
lstm_autoencoder.compile(optimizer='adam', loss='mse')
lstm_autoencoder.summary()

# %%
# Train the model
history = lstm_autoencoder.fit(
    train_data, train_data, 
    epochs=60, 
    batch_size=32, 
    validation_data=(val_data,val_data),
    shuffle=True
)

# %%
plt.figure(figsize=(12, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.grid(True)
plt.legend()

# %% [markdown]
# ## Reconstruction of normal data
# Let's first examine how well the model reconstructs the normal data before attempting to detect anomalies in the abnormal signals.

# %%
def LSTM_reconstruction(autoencoder, test_data,N, M, sample_index):
    # Reconstruct the data   
    decoded_data = autoencoder.predict(test_data)
    decoded_data = decoded_data.reshape(N, M)
    t = np.arange(0,M,1)
    test_data = test_data.reshape(N,M)

    
    # Plot the original and reconstructed data
    plt.figure(figsize=(12, 4))
    plt.plot(t, test_data[sample_index], 'b', label="Input")
    plt.plot(t, decoded_data[sample_index], 'r', label="Reconstruction")
    plt.fill_between(np.arange(len(test_data[sample_index])), decoded_data[sample_index], test_data[sample_index], color='lightcoral', label="Error")
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('LSTM AE', size =14)
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.show()

# %%
for i in range(1,5):
    LSTM_reconstruction(lstm_autoencoder, X_val, N=809, M=180, sample_index=i)

# %% [markdown]
# By using an LSTM architecture, the model successfully identified rare peaks that can appear in normal data, which is crucial for this problem. Next, let’s see if we can detect anomalies in abnormal data.

# %% [markdown]
# ## Detecting abnormal data

# %% [markdown]
# Let's evaluate how our model reconstructs the abnormal data. The expectation is that the model will not reconstruct abnormal data well. By examining the reconstruction error, we can establish a threshold. If the error exceeds this threshold, the data point is classified as an anomaly

# %%
for i in range(1,5):
    LSTM_reconstruction(lstm_autoencoder, X_test, N=X_test.shape[0], M=180, sample_index=i)

# %% [markdown]
# ## Threshold definition

# %%
def error_threshold(autoencoder, train_data, N, M):
    # Reconstruct the training data
    train_predictions = autoencoder.predict(train_data)
    train_predictions = train_predictions.reshape(N, M)
    train_data = train_data.reshape(N, M)
    
    # Calculate the train MSE
    train_errors = np.square(train_data - train_predictions).mean(axis=1)
    error= np.square(train_data - train_predictions).flatten()
    # Define the threshold for anomaly detection
    threshold = train_errors.mean() + 8*train_errors.std()
    
    # Plot the error distribution for the train data
    plt.figure(figsize=(8, 4))
    plt.hist(error, bins=200, alpha=0.75, color='blue', edgecolor='black',density=True)
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Validation Error Distribution')
    plt.xlim([None,0.10])
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return threshold

# %%
threshold = error_threshold(lstm_autoencoder, X_val, N=809, M=180)

# %%
def LSTM_anomaly_detection(autoencoder, test_data, threshold, N, M, sample_index):
    # Reconstruct the data   
    decoded_data = autoencoder.predict(test_data)
    decoded_data = decoded_data.reshape(N, M)
    t = np.arange(0,M,1)
    test_data = test_data.reshape(N,M)
    
    # Calculate mean squared error between the test data and reconstructed data
    val_mse = np.square(test_data - decoded_data)
    
    # Identify anomalies (boolean matrix)
    anomalies = val_mse > threshold
    
    # Check if there are any anomalies in the series
    if np.any(anomalies[sample_index]):
        print("Anomalies detected in the series.")
    else:
        print("No anomalies detected in the series.")
    
    # Plot the original and reconstructed data for a sample
    plt.figure(figsize=(12, 4))
    plt.plot(t, test_data[sample_index], 'b', label="Input")
    plt.plot(t, decoded_data[sample_index], 'r', label="Reconstruction")
    
    # Highlight anomalies
    anomaly_indices = np.where(anomalies[sample_index])[0]
    if anomaly_indices.size > 0:
        plt.scatter(anomaly_indices, test_data[sample_index][anomaly_indices], color='orange', s=40, label='Anomaly')
    
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('LSTM AE with Anomalies', size=14)
    plt.legend()
    plt.grid(True)
    plt.show()

# %%
for i in range(1,5):
    LSTM_anomaly_detection(lstm_autoencoder, X_test, threshold = threshold, N=X_test.shape[0], M=180, sample_index=i)

# %%
for i in range(1,5):
    LSTM_anomaly_detection(lstm_autoencoder, X_val, threshold = threshold, N=X_val.shape[0], M=180, sample_index=i)

# %% [markdown]
# We have successfully defined a threshold that accurately identifies anomalies in the abnormal data, with only a few false positives among the normal data. Additionally, the model can be further calibrated to ensure that isolated points do not erroneously indicate an anomaly. Moreover, it is important to note that the model's hyperparameters have not yet been fine-tuned, leaving room for potential improvements in performance.


