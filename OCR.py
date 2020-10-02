from PIL import Image
import pytesseract

import numpy as np
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr(path):
    img = Image.open(path)

    #norm_img = np.zeros((img.shape[0], img.shape[1]))
    #img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    #img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    #img = cv2.GaussianBlur(img, (1, 1), 0)

    # select language eng+deu
    # select oem --oem 1 for LSTM --oem 0 for Legacy Tesseract
    # select psm 3

    text = pytesseract.image_to_string(img)
    
    return text