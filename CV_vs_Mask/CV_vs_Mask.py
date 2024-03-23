# -*- coding: utf-8 -*-
import cv2
import numpy as np

def extract_white_squares(image_path):
    # �������� �����������
    image = cv2.imread(image_path)
    # �������������� � ����� ����
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ���������� ������ ��� ������ ����� ��������
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # ����� �������� �� �����������
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    squares = []  # ������ ��� ���������� ��������� ������� ���������

    for cnt in contours:
        # ��������������� ������������� �������
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        
        # �������� �� ������� (4 �������) � �� ����������� �������, ����� ��������� ������ �������
        if len(approx) == 4 and cv2.contourArea(cnt) > 1000:  
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # ���������� ������� � ������ � ������ squares
            roi = image[y:y+h, x:x+w]
            squares.append(roi)

    # ����� �������� � ������������� ������ ����������
    for idx, square in enumerate(squares):
        cv2.imshow(f'Square {idx+1}', square)
    
    # ����� ������ ���������� ��������� ���������
    print(f"Total white squares found: {len(squares)}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# �������� ���� � ����� �����������
image_path = r'D:\VS Projects\CV_vs\CV_vs\test\TEST2.jpg'
extract_white_squares(image_path)

