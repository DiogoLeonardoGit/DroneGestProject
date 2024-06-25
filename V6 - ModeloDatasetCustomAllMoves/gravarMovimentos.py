import os
import serial
import time
import shutil
from datetime import datetime

# Abre a porta serial
arduino_port = 'COM5'
baud_rate = 115200
ser = serial.Serial(arduino_port, baud_rate, timeout=1)  # Adiciona timeout de 1 segundo

# Pede informações ao usuário
file_name = input("Digite o nome do arquivo: ")
gesture_id = int(input("Digite o ID do movimento (de 0 a 9): "))

# Define o cabeçalho do arquivo CSV
header = "Group_id,Time(ms),AccelX,AccelY,AccelZ,GyroX,GyroY,GyroZ\n"

# Define o caminho do arquivo
file_path = 'dataset/' + file_name + '.csv'
validation_file_path = 'dataset/validation_' + file_name + '.csv'

# Verifica se o arquivo já existe
file_exists = os.path.exists(file_path)
validation_file_exists = os.path.exists(validation_file_path)

# Obter a data e hora atual
now = datetime.now()
formatted_date = now.strftime("%d-%m-%Y_%H%M%S")

# se existirem, duplicar os ficheiros antes de usar para ficar como backup
if file_exists and validation_file_exists:
    shutil.copyfile(file_path, "backups/" + file_path + "_" + formatted_date + "_backup.csv")
    shutil.copyfile(validation_file_path, "backups/" + validation_file_path + "_" + formatted_date + "_backup.csv")
    print("Backup Files Created.")


# Abre o arquivo no modo de adição ('a' para adicionar) ou cria um novo se não existir
with open(file_path, 'a' if file_exists else 'w') as file, \
     open(validation_file_path, 'a' if validation_file_exists else 'w') as validation_file:

    # Se o arquivo não existir, escreve o cabeçalho
    if not file_exists:
        file.write(header)

    # Inicializa o group_id com base no número da última iteração
    if file_exists:
        with open(file_path, 'r') as f:
            last_line = f.readlines()[-1]
            last_group_id = float(last_line.split(',')[0])
            group_id = last_group_id + 1
    else:
        group_id = 1

    # Escreve no arquivo de validação apenas se estiver vazio
    if not validation_file_exists or os.path.getsize(validation_file_path) == 0:
        validation_file.write("Group_id,Gesture_id\n")

    # Define variáveis de controle
    movements_count = 0
    max_movements = 100

    # inicio da gravacao
    print("Prepare-se para gravar os movimentos... A começar em 3 segundos...", end='')
    time.sleep(1)
    print(" 2...", end='')
    time.sleep(1)
    print(" 1...", end='')
    time.sleep(1)
    print("VAI")

    try:
        while movements_count < max_movements:
            # Registra o movimento por 2 segundos
            print(f"[{movements_count}] A registar movimento {gesture_id} por 2 segundos...")
            start_time = time.time()

            # Defina o número de amostras desejado para 2 segundos
            samples_per_period = 64
            samples_count = 0
            movement_data = ""

            # Clear the serial input buffer to get the most recent data
            ser.reset_input_buffer()

            while samples_count < samples_per_period:
                line = ser.readline().decode('utf-8').strip()  # Lê uma linha da porta serial
                print("lido: " + line)
                if line != 'MPU6050 Connected!':  # Verifica se a linha não está vazia
                    data = line.split(',')
                    if len(data) >= 6:
                        accel_x = data[1]
                        accel_y = data[2]
                        accel_z = data[3]
                        gyro_x = data[4]
                        gyro_y = data[5]
                        gyro_z = data[6]
                        current_time = int(round(time.time() * 1000))
                        movement_data += f"{group_id},{current_time},{accel_x},{accel_y},{accel_z},{gyro_x},{gyro_y},{gyro_z}\n"
                        samples_count += 1

                # Pause para manter a taxa de amostragem
                time.sleep(1 / 32)  # 32 amostras por segundo

            # Escreve nos ficheiros de data e validação
            file.write(movement_data)
            file.flush()

            validation_file.write(f"{group_id},{gesture_id}\n")
            validation_file.flush()

            movements_count += 1
            group_id += 1

            # Pausa por 3 segundos entre os movimentos
            if movements_count < max_movements:
                print(f"[{movements_count}] Pausando por 2 segundos... 2...", end='')
                time.sleep(1)
                print(" 1...", end='')
                time.sleep(1)
                print("VAI")


        print("Todos os movimentos foram registrados.")
    except KeyboardInterrupt:
        # perguntar se quer guardar os dados
        print("Dados Guardados")
