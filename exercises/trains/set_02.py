import sys
sys.path.append('.')
from utils import load, show, show_many
import cv2
import matplotlib.pyplot as plt

img = load('assets/objects_shapes.png')
# show(img, 'betoche')

img2 = cv2.imread('assets/objects_shapes.png')
# print(img2.min())
# print(img2[:, :, 0].max())
# cv2.imshow('betocheh', img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print("OpenCV (B, G, R) : ", img[90, 100])
# print("Matplot (R, G, B) : ", rgb[90, 100])
# plt.figure(figsize=(6,4))
# plt.imshow(rgb)
# plt.title('wrong: colours, red circles look blue!')
# # plt.axis('off')
# plt.show()
corner = img[:, 0:100]
# print("corner shape", corner.shape)
# show(corner, 'img[0:50, 0:100] - the top left corner')

# pixel = img[90, 100]
# blue, green, red = pixel
# print("blue = ", blue, " green = ", green, " red = " , red)

# edited = img.copy()
# edited[10:40, 250:350] = (0, 255, 255)
# edited[100, 100] = (255, 255, 255)
# show(edited, 'edited copy')

blue , green , red = cv2.split(img)
print("one channel shape: ", red.shape)
show_many([blue, green, red],['blue channel', 'green channel', 'red channel'], columns=3, size=(12,3))

