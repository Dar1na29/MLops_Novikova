import os
import csv

# путь к папке с данными
data_folder = "E:/Study/MLOps/LABS/MLops_Novikova/manifest"

# список файлов в папке
files = os.listdir(data_folder)

# создание индексного файла в формате CSV и запись в него путей к файлам
with open("index.csv", "w", newline="") as index_file:
    writer = csv.writer(index_file)
    for file in files:
        # создание пути к файлу
        file_path = os.path.join(data_folder, file)
        # запись пути в индексный файл
        writer.writerow([file_path])

# сохранение файла на диск
    file_path = "E:/Study/MLOps/LABS/MLops_Novikova/index.csv"
    with open(file_path, "w") as f:
        with open("index.csv", "r") as index_file:
            f.write(index_file.read())