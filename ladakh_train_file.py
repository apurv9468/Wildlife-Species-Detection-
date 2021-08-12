#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# set the matplotlib backend so figures can be saved in the background
import matplotlib
get_ipython().run_line_magic('matplotlib', 'notebook')
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import imutils
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import AveragePooling2D 
from keras.layers import Input
from keras.models import Model
import time
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from keras.models import load_model


# In[ ]:


import split_folders
##split_folders.ratio('/home/ubuntu/Ladakh', output="/home/ubuntu/Ladakh_splitted", seed=1000, ratio=(.8, .2)) # default value


# In[ ]:


train_path= r"/home/ubuntu/Ladakh_splitted/train"
val_path  =   r"/home/ubuntu/Ladakh_splitted/val"


# In[ ]:


totalTrain = len(list(paths.list_images(train_path)))
totalVal = len(list(paths.list_images(val_path)))
print("TotalTrain length:",totalTrain)
print("Totalval length:",totalVal)


# In[ ]:


train_aug=ImageDataGenerator( 
                              rotation_range=30,
                              zoom_range=0.2,
                              width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.15,
                             horizontal_flip=True,
                            fill_mode="nearest",
                                   )
## We do not apply Data augmentstaion to valdiation or test set 
## thats why we are passing it  no parameters
val_aug=ImageDataGenerator()

# Mean Values of RGB In IMAGE NET DATASET 
# We will use this mean to subtarcr from the input image for mean subtraction 
mean=np.array([123.68, 116.779, 103.939], dtype="float32")
train_aug.mean=mean
val_aug.mean=mean


# In[ ]:


batch_size=64
classes=19


# In[ ]:



## resnet 50 input shape is  224 *224
train_data_gen=train_aug.flow_from_directory(train_path,class_mode='categorical',target_size=(224,224),
                                             color_mode="rgb",shuffle=True,batch_size=batch_size)

val_data_gen=val_aug.flow_from_directory(val_path,class_mode='categorical',target_size=(224,224),
                                          color_mode="rgb",shuffle=True,batch_size=batch_size)  


# print("[INFO] preparing model...")
# baseModel = ResNet50(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# 
# headModel = Dense(512, activation="relu",kernel_regularizer=regularizers.l2(1e-2), 
#                                          bias_regularizer=regularizers.l2(1e-2))(headModel)
# 
# headModel = Dropout(0.5)(headModel)
# 
# headModel = Dense(256, activation="relu",kernel_regularizer=regularizers.l2(1e-2), 
#                                          bias_regularizer=regularizers.l2(1e-2))(headModel)
# 
# headModel = Dropout(0.5)(headModel)
# 
# headModel = Dense(19,activation='softmax',kernel_regularizer=regularizers.l2(1e-2), 
#                                           bias_regularizer=regularizers.l2(1e-2))(headModel)
# ladakh_model =Model(inputs=baseModel.input,outputs=headModel)
# 
# 
# 

# In[ ]:


## importing VGG16 and loading the weights of Image Net Data set:

baseModel=VGG16(weights='imagenet',include_top=False,input_tensor=Input(shape=(224, 224, 3)))


# In[ ]:


headmodel=baseModel.output
##headmodel= AveragePooling2D(pool_size=(7, 7))(headmodel)
headmodel=Flatten(name="flatten")(headmodel)

## Applying regulariztion in dense layers to  prevent overfitting during the time of training of the head of the model

headmodel=Dense(256,activation="relu",  kernel_regularizer=regularizers.l2(1e-2), 
                                        bias_regularizer=regularizers.l2(1e-2))(headmodel)

headmodel=Dropout(0.5)(headmodel)


## Applying regulariztion in dense layers to  prevent overfitting during the time of training of the head of the model

headmodel=Dense(classes,activation="softmax", kernel_regularizer=regularizers.l2(1e-2), 
                                              bias_regularizer=regularizers.l2(1e-2))(headmodel)

ladakh_model=Model(inputs=baseModel.input,outputs=headmodel)


# In[ ]:


## Freexing the top Base model layers  to prevent them from training at start
for layer in baseModel.layers:
    layer.trainable = False
for layer in ladakh_model.layers:
    print("{}: {}".format(layer, layer.trainable))
    
    
        


# In[ ]:


EPOCHS = 50
INIT_LR=1e-4
start = time.time()

print("[INFO] compiling model...")
opt=Adam(learning_rate=INIT_LR,beta_1=0.9,beta_2=0.999,decay=INIT_LR / EPOCHS)

## ADAM WAS NOT WORKING WELL SO LETS TRY SGD !

ladakh_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=["accuracy"])


##es=EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
#mc=ModelCheckpoint('best_model_resnet.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
mc_save=ModelCheckpoint('ladakh_model{epoch:08d}.h5', period=5)



print("[INFO] Training the Head of the model...")
H=ladakh_model.fit_generator(train_data_gen,steps_per_epoch=totalTrain/batch_size,epochs=EPOCHS,
                    validation_data=val_data_gen,validation_steps=totalVal/batch_size,callbacks=[mc_save])

print(f'Time: {time.time() - start}')


# In[ ]:


ladakh_model=load_model("ladakh_model00000030.h5",compile=True)


# In[ ]:


for layer in ladakh_model.layers:
    print("{}: {}".format(layer, layer.trainable))


# In[ ]:


for layer in ladakh_model.layers[15:]:
     layer.trainable = True


# In[ ]:


for layer in ladakh_model.layers:
    print("{}: {}".format(layer, layer.trainable))


# In[ ]:


train_data_gen.reset()
val_data_gen.reset()


# In[ ]:


EPOCHS = 20
INIT_LR = 1e-5


print("[INFO]  Re compiling model...")
opt=Adam(learning_rate=INIT_LR,beta_1=0.9,beta_2=0.999,decay=INIT_LR / EPOCHS)
ladakh_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=["accuracy"])


print(" [INFO]  Re_Training the Head  plus last  block of convolution in  the model...")
H=ladakh_model.fit_generator(train_data_gen,steps_per_epoch=totalTrain/batch_size,epochs=EPOCHS,
                    validation_data=val_data_gen,validation_steps=totalVal/batch_size)


# In[ ]:


ladakh_model.save("ladakh_model_finetuned.h5", include_optimizer=True)


# In[ ]:


ladakh_model=load_model("ladakh_model_finetuned.h5",compile=True)


# In[ ]:


for layer in ladakh_model.layers:
    print("{}: {}".format(layer, layer.trainable))


# In[ ]:


EPOCHS = 20
INIT_LR = 1e-5


print("[INFO]  Re compiling model...")
opt=Adam(learning_rate=INIT_LR,beta_1=0.9,beta_2=0.999,decay=INIT_LR / EPOCHS)
ladakh_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=["accuracy"])

mc_save=ModelCheckpoint('ladakh_model{epoch:03d}.h5', period=5)


print(" [INFO]  Re_Training the Head  plus last  block of convolution in  the model...")
H=ladakh_model.fit_generator(train_data_gen,steps_per_epoch=totalTrain/batch_size,epochs=EPOCHS,
                    validation_data=val_data_gen,validation_steps=totalVal/batch_size,callbacks=[mc_save])


# In[ ]:


ladakh_model=load_model("ladakh_model_new005.h5",compile=True)


# In[ ]:


EPOCHS = 10
INIT_LR = 1e-4


print("[INFO]  Re compiling model...")

opt=SGD(learning_rate=INIT_LR, momentum=0.9)
ladakh_model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=["accuracy"])

mc_save=ModelCheckpoint('ladakh_model_new_le{epoch:03d}.h5', period=5)


print(" [INFO]  Re_Training the Head  plus last  block of convolution in  the model...")
H=ladakh_model.fit_generator(train_data_gen,steps_per_epoch=totalTrain/batch_size,epochs=EPOCHS,
                    validation_data=val_data_gen,validation_steps=totalVal/batch_size,callbacks=[mc_save])


# In[ ]:




