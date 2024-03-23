# -*- coding: utf-8 -*-
import cv2
import pytesseract
import numpy as np
from Functions import PreprocessImg

def find_and_draw_boxes(frame, config):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    details = pytesseract.image_to_data(frame_rgb, output_type=pytesseract.Output.DICT, config=config)

    for i, word in enumerate(details['text']):
        if word.isalnum() and details['conf'][i] > 30:
            x, y, w, h = details['left'][i], details['top'][i], details['width'][i], details['height'][i]
            if w*h > 10:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, word, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print(word)
    return frame

def process_image(frame, custom_config):
    preprocessed = PreprocessImg.preprocess_image(frame)
    height, width = preprocessed.shape
    scale_factor = 1.5
    preprocessed_scaled = cv2.resize(preprocessed, (int(width * scale_factor), int(height * scale_factor)))
    preprocessed_scaled_color = cv2.cvtColor(preprocessed_scaled, cv2.COLOR_GRAY2BGR)
    annotated_frame = find_and_draw_boxes(preprocessed_scaled_color, custom_config)
    annotated_frame_resize_back = cv2.resize(annotated_frame, (int(width), int(height)))
    return annotated_frame_resize_back