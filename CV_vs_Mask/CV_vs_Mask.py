# -*- coding: utf-8 -*-
import cv2
import numpy as np

def extract_white_squares(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    # Преобразование в серый цвет
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Применение порога для поиска белых областей
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Поиск контуров на изображении
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []  # Список для сохранения координат центров квадратов

    for cnt in contours:
        # Приблизительная аппроксимация контура
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        
        # Проверка на квадрат (4 вершины) и на достаточную площадь, чтобы исключить мелкие контуры
        if len(approx) == 4 and cv2.contourArea(cnt) > 1000:  
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Добавление области с буквой в список squares
            roi = image[y:y+h, x:x+w]
            squares.append(roi)

    # Вывод участков с обнаруженными белыми квадратами
    for idx, square in enumerate(squares):
        cv2.imshow(f'Square {idx+1}', square)
    
    # Вывод общего количества найденных квадратов
    print(f"Total white squares found: {len(squares)}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Указание пути к файлу изображения
image_path = r'D:\VS Projects\CV_vs\CV_vs\test\TEST2.jpg'
extract_white_squares(image_path)

