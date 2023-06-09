import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

data_dir = "E:\Study\MLOps\LABS\MLops_Novikova"
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "valid")

train_data = []
for subdir in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, subdir)
    for filename in os.listdir(class_dir):
        img = load_img(os.path.join(class_dir, filename), target_size=(224, 224))
        x = img_to_array(img) / 255.
        train_data.append((x, subdir))

valid_data = []
for subdir in os.listdir(valid_dir):
    class_dir = os.path.join(valid_dir, subdir)
    for filename in os.listdir(class_dir):
        img = load_img(os.path.join(class_dir, filename), target_size=(224, 224))
        x = img_to_array(img) / 255.
        valid_data.append((x, subdir))

train_data = {
    'inputs': np.array(x),
    'targets': np.array(subdir),
}

valid_data = {
    'inputs': np.array(x),
    'targets': np.array(subdir),
}

with open(os.path.join(data_dir, "data", "train.pkl"), "wb") as f:
    pickle.dump(train_data, f)

with open(os.path.join(data_dir, "data", "valid.pkl"), "wb") as f:
    pickle.dump(valid_data, f)
