# Классификация изображений с помощью GLCM матрицы
**Нейросеть прямого распространения** должна классифицировать болезнь пшеницы. Для решения данной задачи каждое изображение разделяется на 4 канала (красный, синий, зеленый, черно-белый). Далее на каждый канал строится GLCM матрица по 5 расстояниям и 4 углам. Из каждой матрице выделяется 6 признаков (контраст, непохожесть, однородность, ASM, энергия, корреляция). То есть на выходе получается вектор из 480 элементов.

Далее вектор заходит в нейросеть прямого распространения:
1. 480 входов
2. 726 нейронов, ReLU (дропаут 0.05)
3. 512 нейронов, ReLU (дропаут 0.05)
4. 264 нейрона, ReLU
5. 42 нейрона, ReLU
6. 3 выхода, softmax


Ошибка вычисляется категориальной кросс-энтропией. Оптимизатор - adam. Нижняя граница обучения - 75%
