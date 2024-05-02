import numpy as np
import pandas as pd
import os
from time import time

from keras.src.saving import load_model
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
le = preprocessing.LabelEncoder()
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization
from keras.optimizers import Adam

import matplotlib.style as style
style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')

import gc
gc.enable()

print("packages loaded")

x_train = pd.read_csv('dataset/X_train.csv')
y_train = pd.read_csv('dataset/y_train.csv')
test = pd.read_csv('dataset/X_test.csv')
print ("Data is ready !!")

"""
Each series has 128 measurements (1 serie = 128 measurements).
It means that each 128 lines (1 serie) = 1 surface
For example, serie with series_id=0 represents a surface = fin_concrete (128 measurements).
"""

# print the shape of the datasets
print("Shape of x_train: ", x_train.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of test: ", test.shape)

# print the unique surfaces
print("Unique surfaces: ", y_train['surface'].unique())

# reshape data to be used in the model
def reshape(data):
    data = data.drop(['row_id', 'series_id', 'measurement_number'], axis=1)
    data = data.values.reshape(-1, 128, 10)
    print("Data reshaped to: ", data.shape)
    return data

x_train = reshape(x_train)
test = reshape(test)

# print the 5 first rows data from x_train
print("x_train data example: ", x_train[0:5])

# reshape the y_train to be used in the model
y_train = y_train['surface']

# print the 5 first rows data from y_train
print("y_train data example: ", y_train[0:5])

# verify if theres any missing values in the datasets
print("Missing values in x_train (nan): ", np.isnan(x_train).any().sum())
print("Missing values in test (nan): ", np.isnan(test).any().sum())

# verify if theres 1 surface from y_train for each serie (128 measurements) in x_train
print("n_surfaces in x_train(x_train rows/128): ", x_train.shape[0] / 128, " == y_train rows: ", y_train.shape[0])

# difine shape to be used in the model
shape = x_train.shape
print("selected shape: ", shape)
print("y_train shape: ", y_train.shape)

# displays a plt of the surfaces count
def plot_surfaces_count():
    plt.figure(figsize=(10, 5))
    sns.countplot(y_train)
    plt.title('Surfaces count')
    plt.show()

plot_surfaces_count()

# encode the y_train to be used in the model
le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train_encoded = le.transform(y_train)
print("Labels encoded: ", y_train_encoded[0:15])

# print the 5 first rows data from y_train_encoded
print("y_train_encoded data example: ", y_train_encoded[0:5])

# split the data to be used in the model
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train_encoded, test_size=0.2, random_state=0)

# print the shape of the datasets
print("Shape of x_train (after reshapes): ", x_train.shape)
print("Shape of y_train (after reshapes): ", y_train.shape)
print("Shape of x_val (after reshapes): ", x_val.shape)
print("Shape of y_val (after reshapes): ", y_val.shape)

if not os.path.exists('myTrainedModel.h5'):
    # create the model
    def create_model():
        model = Sequential()
        model.add(Conv1D(64, 3, activation='relu', input_shape=(shape[1], shape[2])))
        model.add(BatchNormalization())
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Conv1D(256, 3, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(9, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=0.003), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=250, validation_data=(x_val, y_val), verbose=1)

        # save the model
        model.save('myTrainedModel.h5')
        print("Model saved")
        return model, history

    # train the model
    model, history = create_model()

    # plot the accuracy and loss of the model
    def plot_accuracy_loss():
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    plot_accuracy_loss()

else:
    # load the model
    model = load_model('myTrainedModel.h5')
    print("Model loaded")


# Get predicted probabilities for each class
y_pred_probabilities = model.predict(x_val)

# Convert probabilities to predicted class labels
y_pred_classes = np.argmax(y_pred_probabilities, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred_classes)

# print the avg score of the model
print("Avg score: ", accuracy)