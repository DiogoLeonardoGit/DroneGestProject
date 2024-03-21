import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer

# Load the training and test data
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

# Separate features and target variable
train_label = train_data['label']
train_label.head()
trainset = train_data.drop(['label'], axis=1)
trainset.head()

X_train = trainset.values
X_train = trainset.values.reshape(-1, 28, 28, 1)
print(X_train.shape)

test_label = test_data['label']
X_test = test_data.drop(['label'], axis=1)
print(X_test.shape)
X_test.head()

# Standardize the features
lb = LabelBinarizer()
y_train = lb.fit_transform(train_label)
y_test = lb.fit_transform(test_label)

X_test = X_test.values.reshape(-1, 28, 28, 1)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Define the model
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=0,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   shear_range=0,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

X_test = X_test / 255

fig, axe = plt.subplots(2, 2)
fig.suptitle('Preview of dataset')
axe[0, 0].imshow(X_train[0].reshape(28, 28), cmap='gray')
axe[0, 0].set_title('label: 3  letter: C')
axe[0, 1].imshow(X_train[1].reshape(28, 28), cmap='gray')
axe[0, 1].set_title('label: 6  letter: F')
axe[1, 0].imshow(X_train[2].reshape(28, 28), cmap='gray')
axe[1, 0].set_title('label: 2  letter: B')
axe[1, 1].imshow(X_train[4].reshape(28, 28), cmap='gray')
axe[1, 1].set_title('label: 13  letter: M')

plt.show()

# ver se o modelo ja existe
if os.path.exists('TrainedModel.h5'):
    # carregar o modelo
    model = keras.models.load_model('TrainedModel.h5')
    print('Modelo carregado')
else:
    # criar e treinar o modelo

    model = Sequential()
    model.add(Conv2D(128, kernel_size=(5, 5),
                     strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPool2D(pool_size=(3, 3), strides=2, padding='same'))
    model.add(Conv2D(64, kernel_size=(2, 2),
                     strides=1, activation='relu', padding='same'))
    model.add(MaxPool2D((2, 2), 2, padding='same'))
    model.add(Conv2D(32, kernel_size=(2, 2),
                     strides=1, activation='relu', padding='same'))
    model.add(MaxPool2D((2, 2), 2, padding='same'))

    model.add(Flatten())

    model.add(Dense(units=512,activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(units=24,activation='softmax'))
    #model.summary()

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    model.fit(train_datagen.flow(X_train,y_train,batch_size=200),
             epochs = 35,
              validation_data=(X_test,y_test),
              shuffle=1
             )

    (ls,acc)=model.evaluate(x=X_test,y=y_test)
    print('MODEL ACCURACY = {}%'.format(acc*100))

    # Salvar o modelo
    model.save('TrainedModel.h5')


# Fazer previsões nos dados de teste
predictions = model.predict(X_test)

# Converter as previsões em classes
predicted_classes = np.argmax(predictions, axis=1)

# Comparar as previsões com as classes verdadeiras
correct_predictions = (predicted_classes == test_label)

# Imprimir as previsões e se foram corretas ou não
for i in range(len(test_label)):
    print(f'Imagem {i+1}: Classe prevista - {predicted_classes[i]}, Classe verdadeira - {test_label[i]}, Correta? - {correct_predictions[i]}')

# Calcular o número de previsões corretas
num_correct_predictions = np.sum(correct_predictions)

# Calcular o número total de previsões
total_predictions = len(test_label)

# Calcular a porcentagem de acerto
accuracy = (num_correct_predictions / total_predictions) * 100

print(f'Porcentagem de acerto: {accuracy:.2f}%')