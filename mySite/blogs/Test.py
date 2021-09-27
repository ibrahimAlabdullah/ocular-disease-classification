import numpy as np
import matplotlib.pyplot as plt 
import cv2
import tensorflow as tf 
from .Preprocessing import  GaussianBlurfilter,VesselDetection
import os  
from sklearn.feature_selection import SelectFromModel 
import joblib
from django.db import models

IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224
labels = ['Normal', 'Tessellated fundus', 'Pathological myopia', 'CRVO', 'Chorioretinal atrophy-colobom','Vessel tortuosity',
              'Massive hard exudates', 'Severe hypertensive retinopathy', 'Retinitis pigmentosa', 'DR3','Optic atrophy', 'Blur fundus']
    
def TestImage(path): 
    model=tf.keras.models.load_model(r'C:\project\firstProject\mySite\model\model.h5')
    clf=joblib.load(r'C:\project\firstProject\mySite\model\mod.h5')
    mod = SelectFromModel(clf, prefit=True)
    model1=tf.keras.models.load_model(r'C:\project\firstProject\mySite\model\model1.h5')
     
    img_t = cv2.imread(path) 
    img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
    sigma=100 
    img_t=  GaussianBlurfilter(img_t, sigma)
    img_t = cv2.resize(img_t, (IMAGE_WIDTH, IMAGE_HEIGHT)) 
    img_t = VesselDetection(img_t,img_t)
    pred = model.predict(tf.cast(np.array(img_t).reshape(1,IMAGE_WIDTH,IMAGE_HEIGHT,3), tf.float32))
    feature=mod.transform(np.array(pred).reshape(1,pred.shape[1]))
    pr=model1.predict(np.array(feature).reshape(feature.shape[0],feature.shape[1]))  
    maximum=0
    n=0 
    for i in range(len(pr[0])):
        if pr[0][i] > maximum:
            n=i
            maximum=pr[0][i]
    #print("my predict: ",labels[n],"  ",maximum*100,"%")
    return labels[n], round(maximum*100 , 2)
    
 