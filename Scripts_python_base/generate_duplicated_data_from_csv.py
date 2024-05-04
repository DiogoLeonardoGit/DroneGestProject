import os
import shutil
from datetime import datetime

data = "dataset/" + input("Data file: ") + ".csv"
validation_data = "dataset/" + input("Validation data file: ") + ".csv"

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

        # get the last group_id from validation_data file
        last_line = data_file_read.readlines()[-1]
        last_group_id = int(last_line.split(',')[0]) if last_line else 0
        start_number = last_group_id + 1
        copied_group_id = 0

        for i, line in enumerate(data_file_read, start=1):
            # Split the line into parts
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
                    parts2 = line2.strip().split(",")
                    if len(parts2) >= 2:
                        group_id = parts2[0]
                        movement = parts2[1]
                        if int(group_id) == copied_group_id:
                            break

                # Write validation data
                validation_file.write(f"{start_number},{movement}\n")
                start_number += 1
