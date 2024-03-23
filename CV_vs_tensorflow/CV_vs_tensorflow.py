# -*- coding: utf-8 -*-
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

def decode_predictions(scores, geometry, min_confidence):
    # Извлеките высоту и ширину из scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue
            
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    
    return (rects, confidences)

def preprocess_image(frame):
# Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Применяем Гауссово размытие для сглаживания изображения
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Применяем адаптивный порог к размытому изображению
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 10)
    # Морфологические операции для удаления шума
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # Дилатация для лучшего выделения текста
    dilation = cv2.dilate(closing, kernel, iterations=1)
    dilation = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)
    return dilation

def main():
    image_path = r'D:\VS Projects\CV_vs\CV_vs\test\TEST2.jpg'
    east_model_path = r'D:\frozen_east_text_detection.pb-master\frozen_east_text_detection.pb-master\frozen_east_text_detection.pb'

    image = cv2.imread(image_path)
    
    #image = preprocess_image(image)

    orig = image.copy()
    (H, W) = image.shape[:2]

    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    print("[INFO]  EAST text detector...")
    net = cv2.dnn.readNet(east_model_path)

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (rects, confidences) = decode_predictions(scores, geometry, min_confidence=0.5)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # рисуем прямоугольник вокруг текста на исходном изображении
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # после цикла, показываем итоговое изображение с найденными регионами
    cv2.imshow("Text Detection", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
