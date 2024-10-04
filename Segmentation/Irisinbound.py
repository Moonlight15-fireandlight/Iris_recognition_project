import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import diplib as dip
from functions_iris_inbound import extract_adaptive_thershold, applymedianfilter, applyadpthershold, morphological_operations, find_circularity, calculate_centroid, getradiuspupil

def get_pupil(image):

    #image1 = cv.imread('C:/Users/Paul/Desktop/DataSets/CASIA-Iris-Interval-20240212T013505Z-001/CASIA-Iris-Interval/001/L/S1001L01.jpg') #025

    imagemed = applymedianfilter(image, 17)

    histogram, threshold = extract_adaptive_thershold(imagemed)

    imagemed_gray = cv.cvtColor(imagemed,cv.COLOR_BGR2GRAY)

    imagethresholded = applyadpthershold(imagemed_gray, threshold)

    radius1 = 15

    radius2 = 20

    imageclosing = morphological_operations(radius1, radius2, imagethresholded)

    contours, _ = cv.findContours(imageclosing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    area, perimeter, circularity = find_circularity(imageclosing, contours)

    nto = threshold

    new_image_thresholded = 0

    new_imageclosing = 0

    beta = 1/nto

    while circularity < 0.97 : #cambiando al limite de 0.97 para mayor precision para algunos circuos
    
        nto = nto + beta # (To + B) --> To , donde B =1/To
    
        new_image_thresholded = applyadpthershold(imagemed_gray, nto)
    
        new_imageclosing = morphological_operations(radius1, radius2, new_image_thresholded)
    
        contours, _ = cv.findContours(new_imageclosing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # internamente extrae el maximo de las areas
    
        area, perimeter, circularity = find_circularity(new_imageclosing, contours)
    
        if nto == 2*threshold :
        
            #si llega a este momento el nuevo thresold sera el doble y la imagen sera alterada (si hubiera una forma de acercarlo a 
            # su mayor rendimiento)
        
            print("no llego a la circularidad deseada")
        
            break
    
    cX, cY, iris_contour = calculate_centroid(contours)

    radius_pupil = getradiuspupil(iris_contour)

    return cX, cY, radius_pupil


