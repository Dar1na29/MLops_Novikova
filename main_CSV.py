import os
import random
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

data_dir = "E:\Study\MLOps\LABS\MLops_Novikova"

# установить путь к файлам с данными для обучения и валидации
train_file = os.path.join(data_dir, "train.csv")
valid_file = os.path.join(data_dir, "valid.csv")

# установить параметры модели
num_classes = 3  # число классов
input_shape = (224, 224, 3)  # размер входного изображения

print(tf.test.is_built_with_cuda())
# настроить видимые устройства для TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    except RuntimeError as e:
        print(e)

# создать генераторы данных для обучения и валидации
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    pd.read_csv(train_file),
    x_col='path',
    y_col='class',
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_dataframe(
    pd.read_csv(valid_file),
    x_col='path',
    y_col='class',
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical')

# создать модель нейронной сети
if os.path.isfile('animal_classifier2.h5'):
    model = load_model('animal_classifier2.h5')
else:
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

# скомпилировать модель с функцией потерь, оптимизатором и метриками
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# обучить модель
epochs = 1
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    print('-' * 10)
    for step in range(steps_per_epoch):
        x_batch, y_batch = next(train_generator)
        loss, acc = model.train_on_batch(x_batch, y_batch)
        if step % 10 == 0:
            print(f"step {step+1}/{steps_per_epoch}, loss: {loss:.4f}, acc: {acc:.4f}")

    # валидация модели после каждой эпохи
    validation_loss, validation_acc = 0, 0
    for step in range(validation_steps):
        x_batch, y_batch = validation_generator[step]
        loss, acc = model.evaluate(x_batch, y_batch, verbose=0)
        validation_loss += loss
        validation_acc += acc
    validation_loss /= validation_steps
    validation_acc /= validation_steps
    print(f"validation loss: {validation_loss:.4f}, validation acc: {validation_acc:.4f}")

model.save('animal_classifier2.h5')

test_dir = os.path.join(data_dir, "test")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=input_shape[:2],
                                                  batch_size=32,
                                                  class_mode='categorical')
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")