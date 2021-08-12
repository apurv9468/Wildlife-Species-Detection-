#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Making prediction using the trained model.!
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from imutils import paths
import os
import shutil 
from sklearn.metrics  import log_loss
from sklearn.metrics  import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd  
import scikitplot as skplt 
import time
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime 


# In[ ]:


## Pre processing the imput image Before fedding it to the model :
def pre_process(image_path):
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) ## Converting the image into RGB format is the model was trained on RGB Fromat!
    image=cv2.resize(image, (224, 224))
    image=image.astype("float32")
    
    mean = np.array([123.68, 116.779, 103.939], dtype="float32") 
    image=image-mean 
    image=np.expand_dims(image, axis=0)
    return image


model=load_model('ladakh_crossfold_3_20.h5')

##  label map is being generated from train_gen during training  time !!
label_map={'BIRDS': 0,
 'blue sheep': 1,
 'chukar': 2,
 'dogs': 3,
 'domestic': 4,
 'fox': 5,
 'horse': 6,
 'human': 7,
 'ibex': 8,
 'kiang': 9,
 'lynx': 10,
 'marmot': 11,
 'pallas cat': 12,
 'pica': 13,
 'snowcock': 14,
 'snowleopard': 15,
 'stone matin': 16,
 'urial': 17,
 'wolf': 18,
 'woolyhare': 19}
      
label_map =dict(map(reversed, label_map.items()))

def predict_image(image_path,model,topk=5):
        image=pre_process(image_path)
        pred=model.predict(image)[0]
        max_prob_class=np.argmax(pred)
        class_name=label_map[max_prob_class]
        topk_class=(-pred).argsort()[:topk]
        top_k_class_names=[label_map[i] for i in topk_class]
        return class_name,top_k_class_names,max_prob_class
    
    
    


# In[ ]:



## sample path is the input path !!!!!
# input path : path of the folder whoch the users given !!
def run_input(sample_paths):

sample_paths=r'/home/ubuntu/ladakh_test_data_final/val/horse'

    imagePaths=sorted(list(paths.list_images(sample_paths)))

    print("Total Number of images found :",len(imagePaths))
    start = time.time()



    # returns current date and time 
    now = datetime.now() 
    predicted=[]
    for f in imagePaths:

        class_name,top_k_class_names,max_prob_class=predict_image(f, model,topk=1)
        predicted.append(class_name)
        odir = "Output_folder" + '/'+str(now)+'/'+ str(class_name)
        if( not os.path.isdir(odir)):
            os.makedirs(odir)
        shutil.copy(f, odir)  
    print(f'Time: {time.time() - start}')
    output_path= "Output_folder" + '/'+str(now)
    return  output_path

## output folder
#output  folder: is a folder containg sub folder having sub directoires of the predicted animals !


# In[ ]:



run_input(r'/home/ubuntu/ladakh_test_data_final/val/horse')


# In[ ]:




