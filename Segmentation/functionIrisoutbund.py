import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def display_image(img):
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')

def getimagesize(image):

    # Get the dimensions of the image
    height, width, _ = image.shape

    lenx = width - 1
    leny = height - 1

    return lenx, leny

def decimating(image, width, height):
    
    coef_base = np.array([1])
    
    coeff = np.convolve(coef_base, np.array([1, 1]))
    
    coeffs = np.convolve(coeff, np.array([1, 1]))
    
    coeffs = coeffs / np.sum(coeffs)
    
    kernel = np.outer(coeffs, coeffs)
    
    imgsampled = cv.filter2D(image, -1, kernel)
    
    factor = 2
    
    nwidth = width // factor
    nheight = height // factor
    
    decimated_image = cv.resize(imgsampled, (nwidth, nheight))
    
    return kernel, decimated_image


