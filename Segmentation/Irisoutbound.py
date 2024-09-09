import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#from functions_iris_outbound import display_image
from Irisinbound import get_pupil

image1 = cv.imread('C:/Users/Paul/Desktop/DataSets/CASIA-Iris-Interval-20240212T013505Z-001/CASIA-Iris-Interval/012/R/S1012R03.jpg') #025

### Pupil information

pup_center_X, pup_center_Y, pupil_radius = get_pupil(image1)

#print(pupil_radius)

### Create Mask

#pupil_radius = np.round(pupil_radius,0)

pupil_radius = np.round(pupil_radius,0).astype("int")

mask = np.zeros_like(image1)

cv.circle(mask, (pup_center_X, pup_center_Y), radius = pupil_radius + 4, color = (255,255,255), thickness = -1)

### Apply Mask

image2 = cv.bitwise_or(image1, mask)

###

image1gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

imageblur = cv.GaussianBlur(image1gray, (9,9), 6)

edgemap = cv.Canny(imageblur, 5, 30) # (10,30)

(height , width, _) = image1.shape

rmin = int(0.05*width)

rmax = int(0.4*width)

obir = image1.copy()

## Para que sirve el HoughCircles ?
circles = cv.HoughCircles(edgemap, cv.HOUGH_GRADIENT, dp = 1.5, minDist = 250, #100
                           param1 = 100, param2 = 40, minRadius = rmin, maxRadius = rmax) #minRadius=90, maxRadius=110

#print(circles)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int") #
    for (x, y, r) in circles:
        cv.circle(obir, (x, y), r, (0, 255, 0), 2)

#print(circles[0][1])

#xi = circles[0][0]
#yi = circles[0][1]
#ri = circles[0][2]

#print("centro del iris :", "(", xi, ",", yi, ")", "-", "radio del iris :", ri)
print(pup_center_X, pup_center_Y, pupil_radius)

plt.figure(figsize=(8, 4))
plt.subplot(2, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title('Real image')
plt.subplot(2, 2, 2)
plt.imshow(imageblur, cmap='gray')
plt.title("imageblured")
plt.subplot(2, 2, 3)
plt.imshow(edgemap, cmap='gray')
plt.title('Edgemap')
plt.subplot(2, 2, 4)
plt.imshow(obir, cmap='gray')
plt.title('IrisOutbund')
plt.show()


# https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
# https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
# ouptut circles (x, y, radius)

#utilizar el thershold clasico para aplicarlo en un adaptativo, es posible?

