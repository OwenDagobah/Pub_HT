#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd #Import library
import numpy as np #Library for working with arrays
from sklearn import preprocessing
from sklearn.model_selection import train_test_split #Split arrays or matrices into random train and test subsets
from sklearn import linear_model #Machine learning model
from sklearn.metrics import accuracy_score

df = pd.read_csv('sensor_readings_24.csv') #reads from .csv file with
#the Pandas comma separated values to the DataFrame df

input_vars = [] #A list for the input
for i in range (1,25): #iterates trough the values 1-24 
#from the tables first line, headers
    input_vars.append('Sensor'+str(i)) #Makes a list from the headers
    
X = np.array(df[input_vars]) #List without the column names
#y = df['Command'] #A list of the commands
#test = df['Sensor3']

enc = preprocessing.LabelEncoder()
y = enc.fit_transform(np.array(df['Command'])) #fit_transform, allows input on only numbers

#Most of the machine learning algorithms, takes input
#from scaled values.
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X) #Scales the tables original values.
#Average is is 0 and standard deviation is 1.

#New varables
X_train, X_test, y_test, y_train = train_test_split(X_scaled, y, test_size=0.2)
#datasize 20%

#Defining machine learning model
model = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')

#Fit th model according to the given training data
model.fit(X_train, y_train)
#Predict class labels for sambles in X.
df['Predict'] = enc.inverse_transform(model.predict(X_scaled,))

print('Accuracy of the train data', accuracy_score(y_train, model.predict(X_train)))

print('Accuracy of the test data' , accuracy_score(y_test, model.predict(X_test)))

#Confusion matrix
#**************************
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

cm = confusion_matrix(y_test, model.predict(X_test))
plt.figure()
ax = plt.axes()
sn.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt='.0f', ax=ax, 
           xticklabels=enc.inverse_transform([0,1,2,3]), 
           yticklabels=enc.inverse_transform([0,1,2,3]))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=0)
plt.show() 



"""
Created on Sat Sep  5 16:38:21 2020

@author: xxxx
"""
