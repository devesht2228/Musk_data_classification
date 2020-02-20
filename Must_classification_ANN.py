# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:58:39 2020

@author: devesh kumar thakur
"""

# In[ ]: Importing Library

import pandas as pd
import numpy as np

import keras
from keras.models import Model
import keras.layers
from keras import optimizers
import matplotlib.pyplot as plt

import scipy as sp


# Scikit Imports
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split

# pandas display data frames as tables
from IPython.display import display, HTML

import warnings
warnings.filterwarnings('ignore')

import keras
from keras import callbacks
from keras import optimizers
from keras.engine import Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,BatchNormalization
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

# In[ ]: importing Data

df = pd.read_csv("musk_csv.csv")
X = df.iloc[:,3:169]
Y = df.iloc[:,169]
print(X.shape)
print(Y.shape)
# In[ ]: Splitting data into training and testing

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)

print(X_train.shape)
print(X_test.shape)
# In[ ]:

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# In[ ]: ANN model


model = Sequential()

model.add(Dense(600, input_dim=166, kernel_initializer='he_uniform', activation='relu')) 
model.add(BatchNormalization())
model.add(Dropout(.2))

model.add(Dense(500, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())

model.add(Dropout(.2))

model.add(Dense(500, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(.2))

model.add(Dense(500, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(.2))

model.add(Dense(300, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(.2))

model.add(Dense(1, activation='sigmoid')) 
model.summary()

# In[ ]: Compile

model.compile(loss="binary_crossentropy", optimizer="RMSprop", metrics=['accuracy'])
# In[ ]:including Chechpoint


checkpointer = ModelCheckpoint(filepath='checkpoint.hdf5', 
                               verbose=1,save_best_only=True)
# In[ ]: Training 

traind_model = model.fit(X_train, Y_train, epochs = 20,
            callbacks=[checkpointer], batch_size=32, verbose=1,  validation_split=0.2)
# In[ ]:

hist= traind_model.history
print(hist)


# In[123]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(hist["val_accuracy"],label="Validation Accuracy")
plt.plot(hist["accuracy"],label="Training Accuracy")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("Accuracy")
plt.show()


# In[124]:


plt.plot(hist["val_loss"],label="Validation loss")
plt.plot(hist["loss"],label="Training loss")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# In[]: test accurarcy


test_score = model.evaluate(X_test,Y_test)
print(test_score[1])

# In[126]: training accurarcy


train_score = model.evaluate(X_train,Y_train)
print(train_score[1])

# In[128]: saving model

model.save_weights("model_Weights.h5")
model.save("musk_classification")