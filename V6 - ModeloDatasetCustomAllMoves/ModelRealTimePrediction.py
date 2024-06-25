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
arduino_port = 'COM5'
baud_rate = 115200
ser = serial.Serial(arduino_port, baud_rate, timeout=1)  # Adiciona timeout de 1 segundo

################################ record realtime movement ################################
def record_movement():
    samples_per_period = 64
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

    # Reshape the data
    samples_data = samples_data.reshape(1, samples_data.shape[0], samples_data.shape[1])

    # Predict the movement
    prediction = model.predict(samples_data)

    # Format the prediction probabilities as percentages
    formatted_prediction = [round(prob * 100, 2) for prob in prediction[0]]

    return formatted_prediction


################################ record and predict movement ###############################

user_input = ''
while user_input != 'stop':
    # Record the movement
    samples_data = record_movement()

    # Feed the data to the model
    prediction = predict_movement(samples_data)

    # Print the prediction
    print("Prediction:", prediction)

    # moviment per class
    # each position of the prediction array corresponds to the
    # percentage 0-100% of the model's confidence in the (movement) class
    if prediction[0] > 50:
        print("Movimento: Nenhum (noise)")
    elif prediction[1] > 50:
        print("Movimento: Up")
    elif prediction[2] > 50:
        print("Movimento: Down")
    elif prediction[3] > 50:
        print("Movimento: Left")
    elif prediction[4] > 50:
        print("Movimento: Right")
    elif prediction[5] > 50:
        print("Movimento: Back")
    elif prediction[6] > 50:
        print("Movimento: Front")
    elif prediction[7] > 50:
        print("Movimento: Spin")
    elif prediction[8] > 50:
        print("Movimento: Clap")
    elif prediction[9] > 50:
        print("Movimento: Cut")
    else:
        print("Movimento: Não reconhecido")


    user_input = input("Enter para continuar ou 'stop' para parar: ")
