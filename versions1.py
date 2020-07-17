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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os,sys
from scipy import stats
import pandas as pd
import numpy as np


dataset = pd.read_csv('trainData.csv')
dataset.replace('?', np.nan, inplace=True)
#print(dataset["A14"].value_counts())
#replace missing values with high frequency values
dataset = dataset.fillna({"A1": "b"})
dataset = dataset.fillna({"A2": "24.5"})
dataset = dataset.fillna({"A3": "u"})
dataset = dataset.fillna({"A4": "g"})
dataset = dataset.fillna({"A6": "c"})
dataset = dataset.fillna({"A9": "v"})
dataset = dataset.fillna({"A15": "g"})
dataset = dataset.fillna({"A8": "False"})
dataset = dataset.fillna({"A11": "True"})
dataset = dataset.fillna({"A13": "False"})
dataset = dataset.fillna({"A14": "0"})

#convert objects & bool values to int 
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
dataset["A1"] = lb_make.fit_transform(dataset["A1"])
dataset["A2"] = lb_make.fit_transform(dataset["A2"])
dataset["A3"] = lb_make.fit_transform(dataset["A3"])
dataset["A4"] = lb_make.fit_transform(dataset["A4"])
dataset["A6"] = lb_make.fit_transform(dataset["A6"])
dataset["A8"] = lb_make.fit_transform(dataset["A8"])
dataset["A9"] = lb_make.fit_transform(dataset["A9"])
dataset["A11"] = lb_make.fit_transform(dataset["A11"])
dataset["A13"] = lb_make.fit_transform(dataset["A13"])
dataset["A14"] = lb_make.fit_transform(dataset["A14"])
dataset["A15"] = lb_make.fit_transform(dataset["A15"])
dataset["A16"] = lb_make.fit_transform(dataset["A16"])


#replace missing values of integers & floats using mean value of each column 
df_mean_imputed = dataset.fillna(dataset.mean())
df_median_imputed = dataset.fillna(dataset.median())
# print(df_mean_imputed)

array = dataset.values
X = array[:,0:15]
y = array[:,15]
#split the data set
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


#Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_validation = sc_X.transform(X_validation)



# Make predictions on validation dataset
model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
# Evaluate predictions
print('Accuracy: ',accuracy_score(Y_validation, predictions))

print('confusion_matrix: ')
print(confusion_matrix(Y_validation, predictions))

print('classification_report: ')
print(classification_report(Y_validation, predictions))


########################################################################
#for test data set
dataset = pd.read_csv('testdata.csv')
dataset.replace('?', np.nan, inplace=True)
#print(dataset["A14"].value_counts())
#replace missing values with high frequency values
dataset = dataset.fillna({"A1": "b"})
dataset = dataset.fillna({"A2": "24.5"})
dataset = dataset.fillna({"A3": "u"})
dataset = dataset.fillna({"A4": "g"})
dataset = dataset.fillna({"A6": "c"})
dataset = dataset.fillna({"A9": "v"})
dataset = dataset.fillna({"A15": "g"})
dataset = dataset.fillna({"A8": "False"})
dataset = dataset.fillna({"A11": "True"})
dataset = dataset.fillna({"A13": "False"})
dataset = dataset.fillna({"A14": "0"})

#convert objects & bool values to int 
from sklearn.preprocessing import LabelEncoder
lb_make = LabelEncoder()
dataset["A1"] = lb_make.fit_transform(dataset["A1"])
dataset["A2"] = lb_make.fit_transform(dataset["A2"])
dataset["A3"] = lb_make.fit_transform(dataset["A3"])
dataset["A4"] = lb_make.fit_transform(dataset["A4"])
dataset["A6"] = lb_make.fit_transform(dataset["A6"])
dataset["A8"] = lb_make.fit_transform(dataset["A8"])
dataset["A9"] = lb_make.fit_transform(dataset["A9"])
dataset["A11"] = lb_make.fit_transform(dataset["A11"])
dataset["A13"] = lb_make.fit_transform(dataset["A13"])
dataset["A14"] = lb_make.fit_transform(dataset["A14"])
dataset["A15"] = lb_make.fit_transform(dataset["A15"])
# dataset["A16"] = lb_make.fit_transform(dataset["A16"])


#replace missing values of integers & floats using mean value of each column 
df_mean_imputed = dataset.fillna(dataset.mean())
df_median_imputed = dataset.fillna(dataset.median())
# print(df_mean_imputed)

array = dataset.values
X = array[:,0:15]
y = array[:,14]
X = sc_X.transform(X)

predictions = model.predict(X)
# Evaluate predictions
print("Predictions (1-> success & 0-> Failure):")
print( predictions)
