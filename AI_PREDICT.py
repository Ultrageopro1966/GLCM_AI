print("Импорт библиотек")

# Импорт необходимых библиотек
import tensorflow as tf
import keras, cv2, os
import numpy as np
from skimage.feature import graycomatrix, graycoprops

os.system("cls")

# Функция загрузки изображения
def load_img(src:str) -> np.ndarray:
    img = cv2.imread(src)
    blue_img, green_img, red_img = cv2.split(img) # Разбиение изображения на 3 цветовых канала
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Преобразование изображения в черно-белое
    
    # Создание списка признаков текстур для каждого канала. Каждая текстура - матрица, элементы которой соответствуют одному из 4 углов и 5 расстояний
    glcm = [graycomatrix(blue_img, distances=[1, 2, 5, 20, 100], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed=True),
            graycomatrix(green_img, distances=[1, 2, 5, 20, 100], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed=True),
            graycomatrix(red_img, distances=[1, 2, 5, 20, 100], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed=True),
            graycomatrix(gray_img, distances=[1, 2, 5, 20, 100], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed=True)]
    
    # Список признака текстур для выбранного изображения
    signs = []

    # Цикл перебора каналов
    for channel in glcm:
        # Каждая строка ниже преобразует канал по определенному признаку, превращает в вектор и добавляет в список текстур для выбранного изображения
        signs.extend(list(np.ndarray.flatten(graycoprops(channel, "contrast"))))
        signs.extend(list(np.ndarray.flatten(graycoprops(channel, "dissimilarity"))))
        signs.extend(list(np.ndarray.flatten(graycoprops(channel, "homogeneity"))))
        signs.extend(list(np.ndarray.flatten(graycoprops(channel, "ASM"))))
        signs.extend(list(np.ndarray.flatten(graycoprops(channel, "energy"))))
        signs.extend(list(np.ndarray.flatten(graycoprops(channel, "correlation"))))
    return np.array(signs).reshape(1, -1)

# Функция предсказания ответа моделью (на вход принимается путь к файлу)
def predict(src:str) -> int:
    prediction = model.predict(load_img(src))
    return np.argmax(prediction)

# Создание модели нейронной сети с помощью TensorFlow. Нейронов на слоях 726-512-264-42. 3 выхода, 480 входов.
model = keras.Sequential([
    tf.keras.layers.Input(shape=(480,)),
    tf.keras.layers.Dense(726, activation="relu"), # Функция активации - ReLu
    tf.keras.layers.Dropout(0.05), # Выключение части нейронов для поиска общих признаков
    tf.keras.layers.Dense(512, activation="relu"), # Функция активации - ReLu
    tf.keras.layers.Dropout(0.05), # Выключение части нейронов для поиска общих признаков
    tf.keras.layers.Dense(264, activation="relu"), # Функция активации - ReLu
    tf.keras.layers.Dense(42, activation="relu"), # Функция активации - ReLu
    tf.keras.layers.Dense(3, activation="softmax") # Функция активации - Softmax (используется для классификации)
])

# Компилирование модели с оптимизацией градиентого спуска ADAM, категориальной кросс-энтропией (функция вычисления ошибки)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

# Загрузка весов
model.load_weights("weights.h5")
os.system("cls")

# Пример вывода нейронной сети
print(predict("Healthy/Healthy1.jpg"))
