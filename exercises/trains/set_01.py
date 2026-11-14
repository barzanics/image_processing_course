import sys
sys.path.append('.')
from utils import show, load
import numpy as py
import cv2
print('liberaries loaded - move on ... 😍')
letters = load('./assets/letters_clean.png')
# h, w, c = letters.shape
h, w = letters.shape[0], letters.shape[1]
print("The picture is ", w," pixels wide and ", h," pixels tall.")
