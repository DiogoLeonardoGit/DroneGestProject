import os
import serial
import time



# Abre a porta serial
arduino_port = 'COM3'
baud_rate = 115200
ser = serial.Serial(arduino_port, baud_rate, timeout=1)  # Adiciona timeout de 1 segundo

# Pede informações ao usuário
file_name = input("Digite o nome do arquivo: ")
gesture_id = int(input("Digite o ID do movimento (de 1 a 8): "))

# Define o cabeçalho do arquivo CSV
header = "Group_id,Time(ms),AccelX,AccelY,AccelZ,GyroX,GyroY,GyroZ\n"

# Define o caminho do arquivo
file_path = file_name + '.csv'
validation_file_path = 'validation_' + file_name + '.csv'

# Verifica se o arquivo já existe
file_exists = os.path.exists(file_path)
validation_file_exists = os.path.exists(validation_file_path)

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
            last_group_id = int(last_line.split(',')[0])
            group_id = last_group_id + 1
    else:
        group_id = 1

    # Escreve no arquivo de validação apenas se estiver vazio
    if not validation_file_exists or os.path.getsize(validation_file_path) == 0:
        validation_file.write("Group_id,Gesture_id\n")

    # Define variáveis de controle
    movements_count = 0
    max_movements = 100

    while movements_count < max_movements:
        # Registra o movimento por 2 segundos
        print(f"Registrando movimento {gesture_id} por 2 segundos...")
        start_time = time.time()
        while time.time() - start_time < 2:
            line = ser.readline().decode('utf-8').strip()  # Lê uma linha da porta serial
            if line:  # Verifica se a linha não está vazia
                data = line.split(',')
                if len(data) >= 6:
                    accel_x = data[0].split(':')[1]
                    accel_y = data[1].split(':')[1]
                    accel_z = data[2].split(':')[1]
                    gyro_x = data[3].split(':')[1]
                    gyro_y = data[4].split(':')[1]
                    gyro_z = data[5].split(':')[1]
                    current_time = int(round(time.time() * 1000))
                    file.write(f"{group_id},{current_time},{accel_x},{accel_y},{accel_z},{gyro_x},{gyro_y},{gyro_z}\n")
                    file.flush()
            time.sleep(0.1)

        print("Registro do movimento concluído.")

        # Escreve no arquivo de validação
        validation_file.write(f"{group_id},{gesture_id}\n")
        validation_file.flush()

        movements_count += 1
        group_id += 1

        # Pausa por 3 segundos entre os movimentos
        if movements_count < max_movements:
            print("Pausando por 3 segundos...")
            time.sleep(3)
            print("Pausa concluída. Registrando próximo movimento...")

    print("Todos os movimentos foram registrados.")
