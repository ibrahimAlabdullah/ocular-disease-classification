# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:20:57 2020

@author: Asus
"""

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import cv2 
from keras.layers.core import Flatten 
import DataController
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D   
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.feature_selection import SelectFromModel 
import joblib

IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224

def VGG_19(x=224,y=224,num_classes=2):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(x,y,3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
 
    model.add(Flatten()) 
    
    return model

 
def plott(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
def train():
    data = []
    labels = []
    classification=[]
    features=[] 
    Epoch=75
    
    data, labels, classification,imagen = DataController.PreprocessingAllImages(IMAGE_WIDTH, IMAGE_HEIGHT)  
    
    num_classes=len(labels)
    X_train,X_test,y_train,y_test = train_test_split(np.array(data),np.array(classification),test_size=0.05)
    
    model=VGG_19(IMAGE_WIDTH,IMAGE_HEIGHT,num_classes)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    
    for i in X_train: 
        p=np.array(i).reshape(1,IMAGE_WIDTH,IMAGE_HEIGHT,3)
        features.append(model.predict(p))
    #model.save(r'C:\Users\Asus\Desktop\Project 2\The Project\model.h5')  
 
    clf = ExtraTreesClassifier(n_estimators = 20 ,criterion ='entropy')
    clf.fit(np.array(features).reshape(len(X_train),features[1].shape[1]), np.array(y_train).reshape(len(X_train),num_classes))
    clf.feature_importances_   
    mod = SelectFromModel(clf, prefit=True)
    SelectFeature = mod.transform(np.array(features).reshape(len(X_train),features[1].shape[1])) 
    #joblib.dump(clf,r'C:\Users\Asus\Desktop\Project 2\The Project\mod.h5')  
    print(SelectFeature.shape)
    print(SelectFeature)
    model1 = Sequential()
    model1.add(Dense(450, activation='relu',input_shape=(SelectFeature[0].shape)))
    model1.add(Dense(num_classes, activation='softmax')) 
    model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 
    model1.summary()
    history1=model1.fit(np.array(SelectFeature) , np.array(y_train).reshape(len(X_train), num_classes), batch_size=50, epochs=Epoch, validation_split=0.1)
    plott(history1)  
    #model1.save(r'C:\Users\Asus\Desktop\Project 2\The Project\model1.h5') 
    
    counter = 0
    Xtest=[] 
    for num_test in range(len(X_test)):  
        pred = model.predict(X_test[num_test].reshape(1,IMAGE_WIDTH,IMAGE_HEIGHT,3))
        feature=mod.transform(np.array(pred).reshape(1,pred.shape[1]))
        pr=model1.predict(np.array(feature).reshape( feature.shape[0],feature.shape[1]))
        Xtest.append(np.array(feature).reshape( feature.shape[0]*feature.shape[1],1)) 
        maximum=0
        n=0 
        for i in range(len(pr[0])):
            if pr[0][i] > maximum:
                n=i
                maximum=pr[0][i] 
     
        for i in range(len(y_test[num_test])):
            if y_test[num_test][i]==1: 
                lab=labels[i]
        if lab== labels[n]:
            counter+=1 
    print(counter ," Correct from ", len(X_test))  
    results = model1.evaluate(np.array(Xtest).reshape(np.array(Xtest).shape[0],np.array(Xtest).shape[1]), y_test, batch_size=50)
    print("test loss: ", results[0])
    print("test acc: ", results[1]) 

#train()