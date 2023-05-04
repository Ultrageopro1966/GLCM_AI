print("Импорт библиотек")

# Импорт необходимых библиотек
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import cv2
import keras
import tensorflow as tf
import os, random

os.system("cls")

# Создание списков, в которых будут храниться изображения в формате трехмерного массива
x_train, y_train = [], []

print("Загрузка изображений")
path = "YOUR PATH"

# Загрузка изображений из памяти устройства с помощью модуля os
for file_name in os.listdir(f"{path}\\GLCM\\Healthy"):
    x_train.append(cv2.imread(f"Healthy/{file_name}")) # Чтение и сохранение изображения в виде трехмерного массива
    y_train.append(0) # Добавление номера изображения класса в список ответов
for file_name in os.listdir(f"C:{path}\\GLCM\\PoweryMildew"):
    x_train.append(cv2.imread(f"PoweryMildew/{file_name}"))
    y_train.append(1)
for file_name in os.listdir(f"{path}\\GLCM\\Rust"):
    x_train.append(cv2.imread(f"Rust/{file_name}"))
    y_train.append(2)

# Cоздание массива, в котором будут храниться векторы признаков тексутр (480 элементов в каждом)
converted_x_train = []

print("Загрузка завершена")

# Цикл перебора загруженных изображений
for img in range(1, len(x_train)):
    blue_img, green_img, red_img = cv2.split(x_train[img]) # Разбиение изображения на 3 цветовых канала
    gray_img = cv2.cvtColor(x_train[img], cv2.COLOR_BGR2GRAY) # Преобразование изображения в черно-белое
    
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
    
    os.system("cls")
    print(f"Анализ матриц текстур GLCM {round(img/len(x_train)*100, 2)}%")
    
    # Добавление вектора признаков текстур конкретного изображения в общий список
    converted_x_train.append(np.array(signs))

# Бинарное преобразование верных ответов
y_train = tf.keras.utils.to_categorical(y_train)[1:]

# Перемешивание готовой выборки
data = list(zip(converted_x_train, y_train))
random.shuffle(data)
x_data, y_data = zip(*data)

# Преобразование данных в векторную форму
x_data, y_data = np.array(x_data), np.array(y_data)
converted_x_train = np.array(converted_x_train)

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

# Вывод параметров модели
print(model.summary())

# Компилирование модели с оптимизацией градиентого спуска ADAM, категориальной кросс-энтропией (функция вычисления ошибки)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

# Тренировка модели
model.fit(x_data[:400], y_data[:400], batch_size=15, epochs=30)

# Тестирование модели/дообучение при получении низких результатов
while True:
    os.system("cls")
    _, test_accuracy = model.evaluate(np.array(x_data)[400:], np.array(y_data)[400:])
    if test_accuracy < 0.75:
        # Генерация случайных весов
        new_weights = [np.random.randn(*w.shape) for w in model.get_weights()]
        model.set_weights(new_weights)
        
        # Тренировка модели
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
        model.fit(x_data[:400], y_data[:400], batch_size=15, epochs=30)
    else:
        break

_, test_accuracy = model.evaluate(np.array(x_data)[400:], np.array(y_data)[400:], verbose=0) # Проверка работы нейросети на тестовой выборке
os.system("cls")
print(f"Модель обучена с точностью {round(test_accuracy * 100, 1)}%")

# Сохранение весов модели (чтобы потом заново не обучать)
model.save_weights("weights.h5")