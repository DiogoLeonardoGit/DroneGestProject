import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Step 1: Read the Data Files
data = pd.read_csv('dataset/dados.csv')

le = preprocessing.LabelEncoder()

# Função para remodelar os dados
def reshape_data(data):
    # Usar o Group_id como classe de atividade
    labels = data['Group_id']
    labels_encoded = le.fit_transform(labels)

    # Remover colunas desnecessárias
    data = data.drop(['Group_id', 'Time (ms)'], axis=1)
    #data = data.values.reshape(-1, 30, 6)
    print("Data reshaped to: ", data.shape)
    return data, labels_encoded


# Remodelar os dados
data, labels_encoded = reshape_data(data)
print("Shape of data:", data.shape)
print("Shape of labels:", labels_encoded.shape)
print("Classes: ", labels_encoded)

# Definir a arquitetura do modelo
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(data.shape[1],)))  # Corrigindo a entrada para Conv1D
model.add(BatchNormalization())
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(256, 3, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(le.classes_), activation='softmax'))  # Camada de saída ajustada para o número de classes

model.compile(optimizer=Adam(learning_rate=0.003), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(data, labels_encoded, epochs=250, validation_split=0.2, verbose=1)

# Avaliar o modelo
loss, accuracy = model.evaluate(data, labels_encoded, verbose=0)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)

# Salvar o modelo
model.save('model.keras')

# Plotar a precisão e a perda
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
