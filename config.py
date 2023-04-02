import os
import csv
import argparse
#!!! python config.py E:\Study\MLOps\LABS\MLops_Novikova\manifest --num_portions 2 !!!

# создание парсера аргументов
parser = argparse.ArgumentParser(description='Create an index file for a folder of data files.')
parser.add_argument('data_folder', type=str, help='path to the folder containing data files')
parser.add_argument('--num_portions', type=int, default=1, help='number of portions of data to select')

# получение аргументов из командной строки
args = parser.parse_args()

# путь к папке с данными
data_folder = args.data_folder

# список файлов в папке
files = os.listdir(data_folder)

# определение размера порции данных
portion_size = len(files) // args.num_portions

# удаление индексных файлов, созданных на предыдущем запуске
for i in range(4):
    if os.path.exists(f"index_{i}.csv"):
        os.remove(f"index_{i}.csv")

# выбор порции данных
start_idx = 0
for i in range(args.num_portions):
    end_idx = min(start_idx + portion_size, len(files))
    portion = files[start_idx:end_idx]

    # создание индексного файла в формате CSV и запись в него путей к файлам
    with open(f"index_{i}.csv", "w", newline="") as index_file:
        writer = csv.writer(index_file)
        for file in portion:
            # создание пути к файлу
            file_path = os.path.join(data_folder, file)
            # запись пути в индексный файл
            writer.writerow([file_path])

    # сохранение файла на диск
    file_path = "E:/Study/MLOps/LABS/MLops_Novikova/indexx.csv"
    with open(file_path, "w") as f:
        with open(f"index_{i}.csv", "r") as index_file:
            f.write(index_file.read())

    start_idx = end_idx


