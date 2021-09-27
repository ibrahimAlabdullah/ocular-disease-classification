import numpy as np
import cv2
import matplotlib.pyplot as plt 
 
def VesselDetection(image,orgImage):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    Clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    ClaheImage = Clahe.apply(image)
    #get kernel ellipse shape
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    #Morphological Transformation
    #open erosion and dilation
    op = cv2.morphologyEx(ClaheImage, cv2.MORPH_OPEN, kernel)
    #close dilation and erosion
    cl = cv2.morphologyEx(op, cv2.MORPH_CLOSE, kernel)

    MorphologyImage = Clahe.apply(cv2.subtract(cl, ClaheImage))
    BloodVessels = NoiseSolving(MorphologyImage,orgImage) 
    return BloodVessels

def NoiseSolving(image,orgImage):
    thresholdValue , thresholdImage = cv2.threshold(image,15,255,cv2.THRESH_BINARY)	
    mask = np.ones(image.shape , dtype="uint8")
    contours, hierarchy = cv2.findContours(thresholdImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) <= 1: 
            cv2.drawContours(mask, [c], -1, 0, -1)
    im = cv2.bitwise_and(image, image, mask=mask)
    thresholdValue ,solveNoise = cv2.threshold(im,15,255,cv2.THRESH_BINARY)			
    contours, hierarchy = cv2.findContours(solveNoise,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
    for c in contours: 
        if cv2.contourArea(c) < 5 :
            cv2.drawContours(mask, [c], -1, 0, -1)	
    blood_vessels = cv2.bitwise_and(solveNoise,solveNoise,mask=mask)	
    i = 0
    j = 0
    for gr, fin in zip(orgImage, blood_vessels):
        for g, f in zip(gr, fin):
            if(f == 255):
                orgImage[i][j] = 0 
            
            j = j + 1
        j = 0
        i = i + 1 
    
    return orgImage	

def GaussianBlurfilter(img, sigmaX):   
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #mask
    height, width, depth = img.shape  
    x = int(width/2)
    y = int(height/2)  
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), min(x,y), 1, thickness=-1)
    #bit wise and
    img = cv2.bitwise_and(img, img, mask= circle_img )
    #convert to gray
    grayImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #extract non zero index
    x, y = np.nonzero(grayImage)
    #crop
    img=img[x.min():x.max(), y.min():y.max(),:]
    #resize
    if img.shape[0]<1000:
        img = cv2.resize(img, (2000,1500))
    #gaussian blur and weight
    img=cv2.addWeighted(img,4,cv2.GaussianBlur( img , (0,0),sigmaX) ,-4 ,30)
    return img 	
 