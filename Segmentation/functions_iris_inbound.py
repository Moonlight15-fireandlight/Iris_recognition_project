import cv2 as cv
import numpy as np
import diplib as dip

def extract_adaptive_thershold(image):
    
    #imed_gray = cv.cvtColor(imagemed, cv.COLOR_BGR2GRAY)
    
    hist = cv.calcHist([image], channels=[0], mask=None, histSize=[256], ranges=[0,256])
    
    non_empty_bins = np.sum(hist != 0) # Cuenta todos los valores diferentes a 0
    
    number_pupil_bins = 0.3*non_empty_bins
    
    number_pupil_bins = int(number_pupil_bins)
    
    grayvalues = []
    
    n = 0
    
    for (grayval,histval) in enumerate(hist):
        if (histval != 0) and (n < number_pupil_bins):
            n = n + 1
            grayvalues = np.append(grayvalues, grayval)
        elif n >= number_pupil_bins:
            break
    
    threshold = np.max(grayvalues) + (1/255)
    
    threshold = np.round(threshold, 3)
    
    return hist, threshold

def applymedianfilter(image, ksize):
    
    #image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    imed1 = cv.medianBlur(image, ksize)
    imed2 = cv.medianBlur(imed1, ksize)
    
    return imed2

def applyadpthershold(image, threshold):
    
    #imed2_gray = cv2.cvtColor(imed2, cv2.COLOR_BGR2GRAY)
    
    (T,threshInv_adaptive) = cv.threshold(image, threshold,  255, cv.THRESH_BINARY_INV) #17, 10
    #(T, threshInv) = cv2.threshold(imed2_gray, 0, 128, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    return threshInv_adaptive

def morphological_operations(radius1 ,radius2, binary_image):
    #radius1 = 2 #10
    kernel01 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2*radius1+1,2*radius1+1))
    #radius2 = 20 #25, #30 para la imagen numero 8
    kernel02 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2*radius2+1,2*radius2+1))
    #morph_fill = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel1, iterations = 1) #3-4para que resulte completamente
    morph_opening = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel01, iterations = 1) #talvez no sea neceraio aplicar
    ###
    morph_fill = cv.morphologyEx(morph_opening, cv.MORPH_CLOSE, kernel02, iterations = 1) # operacion morfologica de llenado (fill)
    
    return morph_fill
    
def find_circularity(image, contours):
    
    # antes de aplicar las operaciones morfologicas 
    # contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    iris_contour = max(contours, key=cv.contourArea)
    area = cv.contourArea(iris_contour)
    
    ###perimeter
    
    labels = dip.Label(image > 0)
    msr = dip.MeasurementTool.Measure(labels, features=["Perimeter"])
    perimeter = []
    for i in range(1,len(contours)+1):
        perimeter = np.append(msr[i]["Perimeter"][0], perimeter) # para ver si hay mas de un area, encontrar el maximo que seria la de la pupila
    
    value_perimeter = np.round(perimeter.max(),2)
    circularity = np.round((4*np.pi*area)/(value_perimeter**2),3)
    
    return area, value_perimeter, circularity
    
def calculate_centroid(contours):
    
    iris_contour = max(contours, key=cv.contourArea)
    
    M = cv.moments(iris_contour)
    
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    
    return cX, cY, iris_contour
    #print(f"Centroid coordinates: ({cX}, {cY})")

#If the pupil object is not found, then it aborts the entire localization process
#if contours
def getradiuspupil(iris_contour):
    
    step = np.zeros(len(iris_contour), dtype = np.int32)

    for i in range(0, len(iris_contour)):
        step[i] = iris_contour[i][0][0]

    X_min = step.min()
    #revisar el codigo de python find
    X_max = step.max()
    
    array_index_min = []
    
    array_index_max = []
    
    for k,j in enumerate(step):
        
        if j == X_min:
            #Cual va ser la condición para escoger el indice correcto para todos los valores de X_min?
            array_index_min = np.append(array_index_min, k)
            
        if j == X_max:
            array_index_max = np.append(array_index_max, k)
            
    ni = int(len(array_index_min)/2)
    
    na = int(len(array_index_max)/2)
    
    index_min = int(array_index_min[ni])
    
    index_max = int(array_index_max[na])
    
    #print(type(index_min), type(index_max))

    Y_min = iris_contour[index_min][0][1] # borrar X_min y obtener la posicion donde se encontro el menor valor de X
    
    Y_max = iris_contour[index_max][0][1] # borrar X_max y obtener la posicion donde se encontro el mayor valor de X
    
    #para esto creo que seria lo correcto utilizar el code find 

    PupilDiameter = np.sqrt((X_max - X_min)**2 + (Y_max - Y_min)**2)

    radiuspupil = 0.5*PupilDiameter # Finally, pupil radius is set as rp=0.2D 
    
    return radiuspupil

