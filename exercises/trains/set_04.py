import sys
import cv2
import numpy as np

IMAGE_PATH = "assets/letters_noisy.png"          # تصویر ورودی (سیاه روی سفید یا سفید روی سیاه)
MIN_AREA = 10                      # حداقل مساحت لکه (برای حذف نویز)
SIZE = 32                          # اندازه‌ی استاندارد قالب‌ها
FONT = cv2.FONT_HERSHEY_SIMPLEX
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def tidy(crop) : 
    """" هر تصویری رو به SIZExSIZE تبدیل می کنه"""
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

def build_templates(): 
    tpl = {}
    for letter in ALPHABET: 
        canvas = np.zeros((120,120), dtype=np.uint8)
        cv2.putText(canvas, letter, (20, 90), FONT, 2.5, 255, 4, cv2.LINE_AA)
        _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(canvas)
        tpl[letter] = tidy(canvas[y:y+h, x: x+w])
    return tpl

