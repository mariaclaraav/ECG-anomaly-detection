import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Tensorflow Keras for CNN, LSTM AutoEncoders
from keras.models import Model, Sequential
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Conv1D, MaxPooling1D, Flatten, Dropout

from sklearn.model_selection import train_test_split
data_path = 'C:/Users/maria/OneDrive/Documentos/Github/ECG-anomaly-detection/data/processed/'
df_normal = pd.read_parquet(data_path + 'ptdb_normal.parquet')
df_anomaly = pd.read_parquet(data_path + 'ptdb_abnormal.parquet')
df_normal['label'] = 0
df_anomaly['label'] = 1

def prepare_autoencoder_data(normal_df, anomaly_df, test_size=0.5, random_state=42):

    # Separating the features and labels of normal data
    X_normal = normal_df.iloc[:, :-1]
    y_normal = normal_df['label']
    
    # Separating the features and labels of anomaly data
    X_anomaly = anomaly_df.iloc[:, :-1]
    y_anomaly = anomaly_df['label']
    
    # Splitting the normal data into training and testing sets
    X_train_normal, X_test_normal, y_train_normal, y_test_normal = train_test_split(
        X_normal, y_normal, test_size=test_size, random_state=random_state)
    
    # Combining the normal test data with the anomaly data to form the final test set
    X_test = pd.concat([X_test_normal, X_anomaly], axis=0)
    y_test = pd.concat([y_test_normal, y_anomaly], axis=0)
    
    # Training set remains only with normal data
    X_train = X_train_normal
    y_train = y_train_normal
    
    # Resetting the indices of the DataFrames
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = prepare_autoencoder_data(df_normal, df_anomaly, test_size=0.2)

print(f'Training set size: {X_train.shape}')
print(f'Test set size: {X_test.shape}')


class detector(Model):
  def __init__(self):
    super(detector, self).__init__()
    self.encoder = tf.keras.Sequential([
                                        layers.Dense(32, activation='relu'),
                                        layers.Dense(16, activation='relu'),
                                        layers.Dense(8, activation='relu')
    ])
    self.decoder = tf.keras.Sequential([
                                        layers.Dense(16, activation='relu'),
                                        layers.Dense(32, activation='relu'),
                                        layers.Dense(140, activation='sigmoid')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

# Build Autoencoder
autoencoder = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(X_train.shape[1], activation='sigmoid')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# Train the Autoencoder
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32,
                validation_split=0.2,
                verbose=0)

# Evaluate on the final test set
reconstructions = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.abs(reconstructions - X_test), axis=1)

# Precision-Recall curve and AUPRC
precision, recall, thresholds = precision_recall_curve(y_test, reconstruction_error)
auprc_test = auc(recall, precision)
print(f'AUPRC on the test set: {auprc_test:.3f}')

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label=f'Test Set (AUPRC = {auprc_test:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Confusion Matrix using the best threshold
best_threshold = thresholds[np.argmax(precision * recall)]
y_pred = (reconstruction_error > best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred, normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Abnormal'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix')
plt.show()
