import time
import numpy as np
import serial
from keras.src.saving import load_model
from sklearn import preprocessing

##################################### Load the model #####################################
# Specify the model to use
model_name = input("Nome do modelo a usar: ")

# load the model
model = load_model(model_name + '.keras')
print("Model loaded")

# Abre a porta serial
arduino_port = 'COM4'
baud_rate = 115200
ser = serial.Serial(arduino_port, baud_rate, timeout=1)  # Adiciona timeout de 1 segundo

################################ record realtime movement ################################
def record_movement():
    samples_per_period = 64
    samples_count = 0
    samples_data = []  # List to store the data

    # inicio da gravacao
    print("Prepare-se para executar o movimento... A começar em 3 segundos...")
    print("3...", end='')
    time.sleep(1)
    print(" 2...", end='')
    time.sleep(1)
    print(" 1...", end='')
    time.sleep(1)
    print("VAI")

    start_time = time.time()

    # Clear the serial input buffer to get the most recent data
    ser.reset_input_buffer()

    for _ in range(samples_per_period):
        line = ser.readline().decode('utf-8').strip()  # Lê uma linha da porta serial
        print("lido: " + line)
        if line != 'MPU6050 Connected!':  # Verifica se a linha não está vazia
            data = line.split(',')
            if len(data) >= 7:  # Changed to 7 to ensure correct indexing
                accel_x = float(data[1])  # Convert to float
                accel_y = float(data[2])  # Convert to float
                accel_z = float(data[3])  # Convert to float
                gyro_x = float(data[4])  # Convert to float
                gyro_y = float(data[5])  # Convert to float
                gyro_z = float(data[6])  # Convert to float
                samples_data.append([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])

        # Pause para manter a taxa de amostragem
        time.sleep(1 / 32)  # 32 amostras por segundo

    print("Movimento registado.")
    return samples_data

################################ model prediction function ###############################
def predict_movement(samples_data):
    # Convert the data to a numpy array
    samples_data = np.array(samples_data)

    # Normalize the data
    samples_data = preprocessing.scale(samples_data)

    # Reshape the data
    samples_data = samples_data.reshape(1, samples_data.shape[0], samples_data.shape[1])

    # Predict the movement
    prediction = model.predict(samples_data)

    return prediction


################################ record and predict movement ###############################

user_input = ''
while user_input != 'stop':
    # Record the movement
    samples_data = record_movement()

    # Feed the data to the model
    prediction = predict_movement(samples_data)
    print("Prediction:", prediction)

    user_input = input("Enter para continuar ou 'stop' para parar: ")