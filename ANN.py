import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("C://Users//DELL//Downloads//Churn_Modelling.csv")

X= df.iloc[:, 3:13]
y= df.iloc[:, 13]

#create dummy variables
geog=pd.get_dummies(X["Geography"], drop_first=True)
gender=pd.get_dummies(X['Gender'], drop_first=True)

#concat the dataframe
X=pd.concat([X,geog,gender], axis=1)

#Drop unnecessary columns
X=X.drop(['Geography', 'Gender'],axis=1)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test=sc.transform(X_test)




#ANN

#importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.layers import Dropout

#initialising the ANN
classifier= Sequential()


#Adding input layer and first hidden layers
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu', input_dim=11))
classifier.add(Dropout(0.3))
#Adding the second hidden layers
classifier.add(Dense(units=6, kernel_initializer='he_uniform', activation='relu'))
classifier.add(Dropout(0.4))
#Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation= 'sigmoid'))


#compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model_history=classifier.fit(X_train,y_train, validation_split=0.33, batch_size=10, epochs =100)




#Predicting the test set results
y_pred= classifier.predict(X_test)
y_pred=(y_pred>0.5)


#calculate accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred, y_test)

