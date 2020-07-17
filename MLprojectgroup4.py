#!/usr/bin/env python
# coding: utf-8



# Check the versions of libraries

# Python version
import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas 
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
import pandas as pd
import numpy as np


#--------------------------------------------------------------------


# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


#----------------------------------------------------------------------



# making data frame from csv file  
data = pd.read_csv("trainData.csv")  

# load test data file 
testdata = pd.read_csv("testdata.csv")  


#----------------------------------------------------------------------


data.replace('?', np.nan, inplace=True)
testdata.replace('?', np.nan, inplace=True)


#----------------------------------------------------------------------


print('print top 15 rows in test dataset')
# head
print(testdata.head(15))


#---------------------------------------------------------------------


#replace missing values with high frequency values
data = data.fillna({"A1": "b"})
data = data.fillna({"A2": "24.5"})
data = data.fillna({"A3": "u"})
data = data.fillna({"A4": "g"})
data = data.fillna({"A6": "c"})
data = data.fillna({"A9": "v"})
data = data.fillna({"A15": "g"})
data = data.fillna({"A8": "False"})
data = data.fillna({"A11": "True"})
data = data.fillna({"A13": "False"})
data = data.fillna({"A14": "0"})

#replace missing values with high frequency values in test dataset
testdata = testdata.fillna({"A1": "b"})
testdata = testdata.fillna({"A2": "24.5"})
testdata = testdata.fillna({"A3": "u"})
testdata = testdata.fillna({"A4": "g"})
testdata = testdata.fillna({"A6": "c"})
testdata = testdata.fillna({"A9": "v"})
testdata = testdata.fillna({"A15": "g"})
testdata = testdata.fillna({"A8": "False"})
testdata = testdata.fillna({"A11": "True"})
testdata = testdata.fillna({"A13": "False"})
testdata = testdata.fillna({"A14": "0"})


#----------------------------------------------------------------------------


print('print top 15 rows in test dataset after filling missing values ')
# head
print(testdata.head(15))


#-----------------------------------------------------------------------------


#Encoding categorial data
labelencoder = LabelEncoder()
data['A1'] = labelencoder.fit_transform(data['A1'])
data['A2'] = labelencoder.fit_transform(data['A2'])
data['A3'] = labelencoder.fit_transform(data['A3'])
data['A4'] = labelencoder.fit_transform(data['A4'])
data['A5'] = labelencoder.fit_transform(data['A5'])
data['A6'] = labelencoder.fit_transform(data['A6'])
data['A7'] = labelencoder.fit_transform(data['A7'])
data['A8'] = labelencoder.fit_transform(data['A8'])
data['A9'] = labelencoder.fit_transform(data['A9'])
data['A10'] = labelencoder.fit_transform(data['A10'])
data['A11'] = labelencoder.fit_transform(data['A11'])
data['A12'] = labelencoder.fit_transform(data['A12'])
data['A13'] = labelencoder.fit_transform(data['A13'])
data['A14'] = labelencoder.fit_transform(data['A14'])
data['A15'] = labelencoder.fit_transform(data['A15'])
data['A16'] = labelencoder.fit_transform(data['A16'])

#Encoding categorial testdata
testdata['A1'] = labelencoder.fit_transform(testdata['A1'])
testdata['A2'] = labelencoder.fit_transform(testdata['A2'])
testdata['A3'] = labelencoder.fit_transform(testdata['A3'])
testdata['A4'] = labelencoder.fit_transform(testdata['A4'])
testdata['A5'] = labelencoder.fit_transform(testdata['A5'])
testdata['A6'] = labelencoder.fit_transform(testdata['A6'])
testdata['A7'] = labelencoder.fit_transform(testdata['A7'])
testdata['A8'] = labelencoder.fit_transform(testdata['A8'])
testdata['A9'] = labelencoder.fit_transform(testdata['A9'])
testdata['A10'] = labelencoder.fit_transform(testdata['A10'])
testdata['A11'] = labelencoder.fit_transform(testdata['A11'])
testdata['A12'] = labelencoder.fit_transform(testdata['A12'])
testdata['A13'] = labelencoder.fit_transform(testdata['A13'])
testdata['A14'] = labelencoder.fit_transform(testdata['A14'])
testdata['A15'] = labelencoder.fit_transform(testdata['A15'])


#----------------------------------------------------------------------


print('print top 15 rows of test dataset after encoding ')
# head
print(testdata.head(15))


#------------------------------------------------------------------------


# Split-out validation dataset
array = data.values
X = array[:,0:15]
y = array[:,15]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=1)



#-------------------------------------------------------------------------


#Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_validation = sc_X.transform(X_validation)
testdata = sc_X.transform(testdata)

# ----------------------------------------------------------------------

# ----------------------------------------------------------------------


print('RandomForest Classifier')
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_validation)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_validation, y_pred))
print("Precision:",metrics.precision_score(y_validation, y_pred))
print("Recall:",metrics.recall_score(y_validation, y_pred))
print('classification_report: ')
print(classification_report(y_validation, y_pred))

a =testpredictions=clf.predict(testdata)
print(testpredictions)

#decode testdata
finaldata = ["Success" if testpredictions > 0.5 else "Failure" for testpredictions in a]
print("Predictions:")
print(finaldata)


# ----------------------------------------------------------------------
#End of the code


#Different models we have tried
"""


print('svm Classifier')
# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(X_train, y_train)
y_pred = model.predict(X_validation)
# Evaluate predictions
print("Accuracy:",accuracy_score(y_validation, y_pred))
print("Precision:",metrics.precision_score(y_validation, y_pred))
print("Recall:",metrics.recall_score(y_validation, y_pred))
print(confusion_matrix(y_validation, y_pred))
print('classification_report: ')
print(classification_report(y_validation, y_pred))


# ----------------------------------------------------------------------


print('KNeighbors Classifier')
#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model using the training sets
knn.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = knn.predict(X_validation)
print("Accuracy:",metrics.accuracy_score(y_validation, y_pred))
print("Precision:",metrics.precision_score(y_validation, y_pred))
print("Recall:",metrics.recall_score(y_validation, y_pred))
print('classification_report: ')
print(classification_report(y_validation, y_pred))



# ----------------------------------------------------------------------


print('Gaussian Classifier')
#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_validation)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_validation, y_pred))
print("Precision:",metrics.precision_score(y_validation, y_pred))
print("Recall:",metrics.recall_score(y_validation, y_pred))
print('classification_report: ')
print(classification_report(y_validation, y_pred))


"""

