import pandas as pd


def get_last_group_id(validation_file):
    # Lê o ficheiro de validação
    df = pd.read_csv(validation_file)
    # Obtém o último group_id
    if not df.empty:
        last_group_id = df.iloc[-1]['Group_id']
    else:
        last_group_id = 0
    return last_group_id


def update_group_id_and_append_data(new_data_file, data_file, validation_file, movement_id):
    # Obtém o último group_id do ficheiro de validação
    last_group_id = get_last_group_id(validation_file)

    # Lê os novos dados
    new_data = pd.read_csv(new_data_file)

    # Atualiza os group_ids no novo conjunto de dados
    new_data['Group_id'] = new_data['Group_id'].apply(lambda x: x + last_group_id)

    # Lê os dados existentes
    existing_data = pd.read_csv(data_file)

    # Junta os novos dados com os dados existentes
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)

    # Lê os dados de validação existentes
    existing_validation_data = pd.read_csv(validation_file)

    # Adiciona uma linha ao ficheiro de validação para cada 64 linhas do novo conjunto de dados
    for i in range(0, len(new_data), 64):
        group_id = new_data.iloc[i]['Group_id']
        validation_row = {'Group_id': group_id, 'Gesture_id': movement_id}
        existing_validation_data = existing_validation_data._append(validation_row, ignore_index=True)

    # Escreve os dados de validação atualizados no ficheiro
    existing_validation_data.to_csv(validation_file, index=False)

    # Escreve os dados atualizados no ficheiro
    updated_data.to_csv(data_file, index=False)

    print("Dados atualizados com sucesso!")

# While para combinar varios ficheiros
dados_file = input("Dataset Original: ") + ".csv"  # Nome do ficheiro com os novos dados
while True:
    novo_dados_file = input("Dataset a copiar: ") + ".csv"   # Nome do ficheiro com os dados existentes
    validation_file = 'validation_' + dados_file # Nome do ficheiro de validação
    movement_id = input("Movement ID: ")  # ID do movimento
    update_group_id_and_append_data(novo_dados_file, dados_file, validation_file, movement_id)
