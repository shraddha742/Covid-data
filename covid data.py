#COVID DATASET
#OBJECTIVE:TO PREDICT DEATHS DUE TO COVID

#DATA COLLECTION
#Import dataset
import pandas as pd
data=pd.read_csv('D:/covid data1.csv')

#DATA CLEANING
#Delete the useless data from dataset
#in this dataset 'S.No' is unwanted
data.drop('S. No.',axis=1,inplace=True)

#DATA ANALYSIS

#COUNTPLOT
#To determine size of deaths
import seaborn as sb
sb.countplot(x='Deaths',data=data)
data['Deaths'].unique()

#PAIRPLOT
#Inorder to see variable-variable relation
sb.pairplot(data,hue='Deaths',height=5)

#CREATE ARRAYS
#x=features
#y=target
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#To convert categorical data to numerical data
from sklearn.preprocessing import LabelEncoder
laben=LabelEncoder()
y=laben.fit_transform(y)
x[:,0]=laben.fit_transform(x[:,0])

#Split universal data to train and test parts
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,train_size=0.7,random_state=42)

#Check size of train and test data
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#Select the algorithm
#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()

#TRAINING ALGORITHM
logreg.fit(x_train,y_train)

#TESTING ALGORITHM
logregacc=logreg.score(x_test,y_test)
print(logregacc)

#PREDICTION
logregpred=logreg.predict(x_test)

#Check number of right vs wrong predictions
from sklearn.metrics import confusion_matrix as cm
conmat = cm(y_test,logregpred)


