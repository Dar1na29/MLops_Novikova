import os
import pandas as pd
import csv

# путь к директории с тренировочными данными
train_dir = "E:\Study\MLOps\LABS\MLops_Novikova\\train"

# список классов
classes = sorted(os.listdir(train_dir))

# записываем данные в файл csv
with open('train.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['path', 'class'])
    for cls in classes:
        for file in os.listdir(os.path.join(train_dir, cls)):
            path = os.path.join(train_dir, cls, file)
            writer.writerow([path, cls])



# путь к директории с валидационными данными
valid_dir = "E:\Study\MLOps\LABS\MLops_Novikova\\valid"

# список классов
classes = sorted(os.listdir(valid_dir))

# записываем данные в файл csv
with open('valid.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['path', 'class'])
    for cls in classes:
        for file in os.listdir(os.path.join(valid_dir, cls)):
            path = os.path.join(valid_dir, cls, file)
            writer.writerow([path, cls])


# загрузка данных из csv-файла
train_df = pd.read_csv('train.csv')
valid_df = pd.read_csv('valid.csv')

# сохранение данных в новые csv-файлы
train_df.to_csv('new_train.csv', index=False)
valid_df.to_csv('new_valid.csv', index=False)