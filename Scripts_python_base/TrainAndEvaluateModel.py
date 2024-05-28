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

filename = input("Enter the dataset filename: ")
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


def evaluate_model(model, x_val, y_val):
    # Get predicted probabilities for each class
    y_pred_probabilities = model.predict(x_val)

    # Convert probabilities to predicted class labels
    y_pred_classes = np.argmax(y_pred_probabilities, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred_classes)

    # print the avg score of the model
    print("Avg score: ", accuracy)

def build_model_name(prefix, EPOCH, BATCH, LEARNING_RATE):
    model_name = prefix + "_" + EPOCH + "_" + BATCH + "_" + LEARNING_RATE + ".keras"
    return model_name

print("Treinar novo modelo ou avaliar um modelo existente? (t/a)")
choice = input()

if choice == 't' or choice == 'T':
    prefix = input("Enter the model prefix name: ")
    MODEL = prefix + ".keras"

    if not os.path.exists(MODEL):

        # input how many models to train
        n_models = int(input("Number of models to train: "))

        if n_models == 1:
            EPOCH = int(input("Epochs: "))
            BATCH = int(input("Batch size: "))
            LEARNING_RATE = float(input("Learning rate: "))
        else:
            EPOCH = [0] * n_models
            BATCH = [0] * n_models
            LEARNING_RATE = [0] * n_models
            for i in range(n_models):
                print("Model ", i + 1)
                EPOCH[i] = int(input("Epochs: "))
                BATCH[i] = int(input("Batch size: "))
                LEARNING_RATE[i] = float(input("Learning rate: "))


        # create the model
        def create_model(EPOCH, BATCH, LEARNING_RATE):
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

            model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            history = model.fit(x_train, y_train, batch_size=BATCH, epochs=EPOCH, validation_data=(x_val, y_val),
                                verbose=1)

            return model, history


        # plot the accuracy and loss of the model
        def plot_accuracy_loss(history):
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


        if n_models == 1:
            # train the model
            model, history = create_model(EPOCH, BATCH, LEARNING_RATE)

            # save the model
            model_name = build_model_name(prefix, str(EPOCH), str(BATCH), str(LEARNING_RATE))
            model.save(model_name)
            print("Model saved")

            # plot the accuracy and loss of the model
            plot_accuracy_loss(history)

            # evaluate the model
            evaluate_model(model, x_val, y_val)

        else:
            for i in range(n_models):
                print("Model ", i + 1)
                # train the model
                model, history = create_model(EPOCH[i], BATCH[i], LEARNING_RATE[i])

                # save the model
                model_name = build_model_name(prefix, str(EPOCH[i]), str(BATCH[i]), str(LEARNING_RATE[i]))
                model.save(model_name)
                print("Model saved")

                # plot the accuracy and loss of the model
                plot_accuracy_loss(history)

                # evaluate the model
                evaluate_model(model, x_val, y_val)

    elif os.path.exists(MODEL):
        # load the model
        model = load_model(MODEL)
        print("Model already exists. Model loaded")

        # evaluate the model
        evaluate_model(model, x_val, y_val)
    else:
        print("Model not found")
        exit()

elif choice == 'a' or choice == 'A':
    MODEL = input("Enter the model full name: ")

    if os.path.exists(MODEL):
        # load the model
        model = load_model(MODEL)
        print("Model loaded")

        # evaluate the model
        evaluate_model(model, x_val, y_val)
    else:
        print("Model not found")
        exit()
else:
    print("Invalid choice")
    exit()
