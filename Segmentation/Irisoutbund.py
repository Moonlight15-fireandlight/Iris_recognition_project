import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from functionIrisoutbund import display_image

image1 = cv.imread('C:/Users/Paul/Desktop/DataSets/CASIA-Iris-Interval-20240212T013505Z-001/CASIA-Iris-Interval/025/L/S1025L02.jpg') #025

image1gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

imageblur = cv.GaussianBlur(image1gray, (9,9), 6)

#display_image(imageblur)

edgemap = cv.Canny(imageblur, 10, 30) # (10,30)

#display_image(edgemap)

(height , width, _) = image1.shape

rmin = int(0.05*width)

rmax = int(0.4*width)

#Hough transform
# Outbund_iris = Obi

obir = image1.copy()

circles = cv.HoughCircles(edgemap, cv.HOUGH_GRADIENT, dp = 1, minDist = 100, #100
                           param1 = 100, param2 = 40, minRadius = rmin, maxRadius = rmax) #minRadius=90, maxRadius=110
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv.circle(obir, (x, y), r, (0, 255, 0), 2)
        
plt.subplot(2, 1, 0)
plt.imshow(image1, cmap='gray')
plt.title('image')
plt.subplot(2,1,1)
plt.imshow(obir, cmap='gray')
plt.title('HoughTransform')
plt.show()

# https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
# https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
# ouptut circles (x, y, radius)

#utilizar el thershold clasico para aplicarlo en un adaptativo, es posible?

