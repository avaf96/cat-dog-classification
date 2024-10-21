# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 14:34:21 2019

@author: Ava Fgh
"""
import glob
import cv2
import numpy as np
from keras.utils import np_utils
from keras import optimizers,losses
from keras.models import load_model
from keras.layers import Dense
from keras.models import Model
from keras.layers import Conv2D,MaxPool2D, Input , Flatten,Layer
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy


Path = 'C:/Users/projects/Cat_Dog_DataSet/Train/'
DataSet =  glob.glob(Path +"*.jpg" ) + glob.glob(Path +"*.jpeg") 
Data = []

for img in DataSet :
    cell = img.split('.')
    cell = cell[0].split('\\')
    
    image = cv2.imread(img)
    image = cv2.resize(image,(100,100));
    image = image / np.max(image)
    image = image.astype(np.float64)
    
    if cell[1] == 'cat' :
        x = {"image" : image ,"class": 0 }
        Data.append(x)
    else:
        x = {"image" : image ,"class": 1 }
        Data.append(x)  
    
train_images = []
train_labels = []

for data in Data :
    train_images.append(data['image'])
    train_labels.append(data['class'])
    
train_images = np.array(train_images)
print(train_images.shape)


model=Model.sequential()

model.add(Layer.Conv2D(64,2,activation='relu',padding='same',input_shape=((100,100,3))))
model.add(Layer.MaxPool2D(pool_size = 2))

model.add(Layer.Conv2D(128,2,activation='relu',padding='same'))
model.add(Layer.MaxPool2D(pool_size = 2))

model.add(Layer.Conv2D(256,2,activation='relu',padding='same'))
model.add(Layer.MaxPool2D(pool_size = 2))

model.add(Layer.Conv2D(256,2,activation='relu',padding='same'))
model.add(Layer.MaxPool2D(pool_size = 2))

model.add(Layer.Flatten())

model.add(Layer.Dense(256,activation = 'relu'))
model.add(Layer.Dense(2,activation = 'softmax'))

model.summary()      
    
model.compile(optimizer=optimizers.adam(lr=1e-4), loss = losses.categorical_crossentropy, metrics=['accuracy','categorical_accuracy'])
history = model.fit(train_images,train_labels,epochs=5,batch_size=128,validation_split=0.1)

model.save('CatAndDog.h5')


#####################Test

test = load_model('C:/Users/projects/Cat & Dog/CatAndDog.h5')

testSet=[]

TestPath = 'C:/Users/projects/Cat_Dog_DataSet/MyTest/'
testDataSet =  glob.glob(TestPath +"*.jpg" )+ glob.glob(TestPath +"*.jpeg")
                          
for img in testDataSet :
    testImg = cv2.imread(img)
    testImg = cv2.resize(testImg,(100,100));
    testImg = testImg / np.max(testImg)
    testImg = testImg.astype(np.float64)
    testSet.append(testImg)    

testSet = np.array(testSet)
PrdTest = test.predict(testSet)
PrdTest= np.argmax(PrdTest,axis=1)
y = []
for z in PrdTest :
    if z == 0 :
        y.append('cat')
    else :
        y.append('dog')

    
    
    