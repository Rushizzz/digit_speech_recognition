import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load and preprocess the dataset
def load_data(dataset_path):
    emotions = []
    features = []

    # Traverse all subdirectories and files
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                # Full path to the file
                file_path = os.path.join(root, file)
                
                # Load the audio file
                signal, sr = librosa.load(file_path, sr=None)

                # Extract features (MFCCs)
                mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
                mfccs_mean = np.mean(mfccs.T, axis=0)

                # Extract emotion label from the file name (assuming RAVDESS naming convention)
                # Adjust this logic if your dataset uses a different convention
                emotion = file.split("-")[2]
                
                features.append(mfccs_mean)
                emotions.append(emotion)

    # Check if we found any files
    if len(features) == 0:
        raise ValueError("No .wav files found in the specified dataset path!")

    return np.array(features), np.array(emotions)

# Encode labels
def encode_labels(labels):
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)
    return labels_encoded, encoder

# Build the RNN model
def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='softmax')  # Assuming 8 emotion classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Plot training results and save as JPG
def plot_results(history):
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Save plot as JPG
    plt.savefig('ser_training_results.jpg', format='jpg', dpi=300)
    plt.show()

# Main script
if __name__ == "__main__":
    # Dataset path
    dataset_path = "ravdess-emotional-speech-audio/versions/1"  # Replace with your dataset path

    # Load and preprocess data
    features, labels = load_data(dataset_path)
    labels_encoded, label_encoder = encode_labels(labels)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

    # Reshape features for LSTM input
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Build the RNN model
    model = build_model(input_shape=(x_train.shape[1], x_train.shape[2]))

    # Train the model
    history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))

    # Save the trained model
    model.save('speech_emotion_recognition_model.h5')

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Plot training results and save the graph
    plot_results(history)
