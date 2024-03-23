# -*- coding: utf-8 -*-
from Functions import ProcessingImg
from Functions import PreprocessImg
import cv2
import pytesseract
import numpy as np
import time
import os

def main():
  pytesseract.pytesseract.tesseract_cmd = r'D:\Programs\Tesseract-OCR\tesseract.exe'
  custom_config = r'--oem 3 --psm 10 -l eng'

  choice = input("Press 'c' for camera or 'f' for folder: ").lower()
  
  if choice == 'f':
    folder_path = r"D:\VS Projects\CV_vs\CV_vs\test"
    
    for filename in os.listdir(folder_path):
      if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(folder_path, filename)
        frame = cv2.imread(image_path)
        char_squares = PreprocessImg.extract_white_squares(frame)
        
        for sq in char_squares:
          annotated_frame = ProcessingImg.process_image(sq, custom_config)
          cv2.imshow('Processed Image', annotated_frame)
          cv2.waitKey(0)  # Wait for key press to move to the next image

    cv2.destroyAllWindows()
    
  elif choice == 'c':
    cap = cv2.VideoCapture(0)
    try:
      while True:
        ret, frame = cap.read()
        if not ret:
          print("Failed to grab frame")
          break
        char_squares = PreprocessImg.extract_white_squares(frame)
        for sq in char_squares:
          annotated_frame = ProcessingImg.process_image(sq, custom_config)
          cv2.imshow('Processed Video Stream', sq)
          time.sleep(1)  # Pause for a short time before capturing the next frame
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
          break
    finally:
      cap.release()
      cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
