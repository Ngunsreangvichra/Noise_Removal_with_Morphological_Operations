import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
img = cv2.imread('smile_noise_2.jpg', cv2.IMREAD_COLOR_RGB)

# Convert to grayscale then binary
img = np.mean(img, axis=2)
img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)[1]

kernel2 = np.ones((2, 2), np.uint8)
kernel3 = np.ones((3, 3), np.uint8)

img1 = cv2.dilate(img, kernel2)
img2 = cv2.erode(img1, kernel3)


plt.imshow(img2, cmap='gray')
plt.show()

