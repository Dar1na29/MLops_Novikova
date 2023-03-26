import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# загрузить обученную модель
model = load_model('animal_classifier.h5')

# установить список классов
class_names = os.listdir('train')

# загрузить изображение и привести его к нужному размеру
img_path = 'E:\Study\MLOps\LABS\MLops_Novikova\\test\\c6.jpg'
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# выполнить классификацию изображения
predictions = model.predict(img_array)
class_idx = np.argmax(predictions[0])
class_name = class_names[class_idx]

# вывести результаты
print(f"Class: {class_name}")
print(f"Probabilities: {predictions[0]}")
