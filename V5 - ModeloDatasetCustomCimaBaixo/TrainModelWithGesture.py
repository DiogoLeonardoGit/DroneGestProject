import os

import numpy as np
import pandas as pd
from keras.src.saving import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

# Step 1: Read the Data Files

filename = input("Enter the file name: ")
data = pd.read_csv('dataset/' + filename + '.csv')
validation = pd.read_csv('dataset/validation_' + filename + '.csv')

le = preprocessing.LabelEncoder()

# Função para remodelar os dados
def reshape_data(data):
    # Extract labels from the 'Group_id' column
    labels = validation['Gesture_id']

    # Drop unnecessary columns
    data = data.drop(['Group_id', 'Time(ms)'], axis=1)

    # Reshape the data into the desired shape
    data = data.values.reshape(-1, 64, 6)

    # encode the labels
    le.fit(labels)
    labels_encoded = le.transform(labels)

    return data, labels_encoded


# Remodelar os dados
data, labels_encoded = reshape_data(data)
print("Shape of data:", data.shape)
print("Shape of labels:", labels_encoded.shape)

# define shape
shape = data.shape

# count the number of unique gestures from the labels_encoded
unique_gestures = len(np.unique(labels_encoded))
print("Number of unique gestures: ", unique_gestures)

def plot_gesture_count():
    # Count the occurrences of each label
    label_counts = pd.Series(labels_encoded).value_counts()

    # plot the gestures count
    plt.figure(figsize=(10, 5))
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.title('Gestures Count')
    plt.xlabel('Gesture')
    plt.ylabel('Count')
    plt.show()

plot_gesture_count()

# split the data to be used in the model
x_train, x_val, y_train, y_val = train_test_split(data, labels_encoded, test_size=0.2, random_state=0)

print("Insira um modelo existente para avaliação ou insira um novo modelo para treinamento")
MODEL = input("Model: ") + ".keras"

if not os.path.exists(MODEL):

    EPOCH = int(input("Epochs: "))
    BATCH = int(input("Batch size: "))
    LEARNING_RATE = float(input("Learning rate: "))

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
        model.add(Dense(unique_gestures, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, batch_size=BATCH, epochs=EPOCH, validation_data=(x_val, y_val), verbose=1)

        # Salvar o modelo

        model.save(MODEL)
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
    model = load_model(MODEL)
    print("Model loaded")


# Get predicted probabilities for each class
y_pred_probabilities = model.predict(x_val)

# Convert probabilities to predicted class labels
y_pred_classes = np.argmax(y_pred_probabilities, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred_classes)

# print the avg score of the model
print("Avg score: ", accuracy)