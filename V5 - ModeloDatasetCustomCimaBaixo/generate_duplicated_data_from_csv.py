import os
import shutil
from datetime import datetime

data = "dataset/" + input("Data file: ") + ".csv"
validation_data = "dataset/" + input("Validation data file: ") + ".csv"
n_generations = int(input("Numero de duplicações (1/movimento): "))

# Verifica se o arquivo já existe
file_exists = os.path.exists(data)
validation_file_exists = os.path.exists(validation_data)

# Obter a data e hora atual
now = datetime.now()
formatted_date = now.strftime("%d-%m-%Y_%H%M%S")


# se existirem, duplicar os ficheiros antes de usar para ficar como backup
if file_exists and validation_file_exists:
    shutil.copyfile(data, "backups/" + data + "_" + formatted_date + "_backup.csv")
    shutil.copyfile(validation_data, "backups/" + validation_data + "_" + formatted_date + "_backup.csv")
    print("Backup Files Created.")


# Open input and output files for appending
with open(data, "a") as data_file, open(validation_data, "a") as validation_file:
    # Assuming you also want to read the existing content for reference
    with open(data, "r") as data_file_read, open(validation_data, "r") as validation_file_read:
        # Find the last non-empty line in data_file_read
        last_line = None
        for line in reversed(list(data_file_read)):
            if line.strip():  # Check if the line is not empty
                last_line = line
                break
        last_group_id = last_line.split(',')[0] if last_line else 0
        start_number = int(last_group_id) + 1
        copied_group_id = 0
        data_file_read.seek(0)

        for i, line in enumerate(data_file_read, start=1):
            # Split the line into parts and skip fist line
            if i == 1:
                continue

            parts = line.strip().split(",")
            if len(parts) >= 7:
                # Extract data
                data = ",".join(parts[1:])
                # Write original group_id
                copied_group_id = int(parts[0])

                # Write original data
                data_file.write(f"{start_number},{data}\n")
            if i % 64 == 0:
                # Extract movement by searching for the group id in the validation_data file
                validation_file_read.seek(0)
                for j, line2 in enumerate(validation_file_read, start=1):
                    if j == 1:
                        continue

                    parts2 = line2.strip().split(",")
                    if len(parts2) >= 2:
                        group_id = parts2[0]
                        movement = parts2[1]
                        if int(group_id) == copied_group_id:
                            break

                # Write validation data
                validation_file.write(f"{start_number},{movement}\n")
                start_number += 1
                n_generations -= 1

            if n_generations == 0:
                break

        print("Data duplicated successfully.")
        exit(0)