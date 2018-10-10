# An instance of NERF
# Using TCGA breast cancer dataset, mRNA=>histological classes
# NERF V0.2.1
# Yue Zhang <yue.zhang@lih.lu>



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
import platform
import time

#

# Read in data and display first 5 rows
B20000 = pd.read_table("P:/VM/TCGA/Data/BRCA/BMReady.txt", sep='\t')

# Creating the dependent variable class
factor = pd.factorize(B20000['histological_type'])
B20000.histological_type = factor[0]
definitions = factor[1]

x = B20000.iloc[:, 1:20530].values   # Features for training
y = B20000.iloc[:, 0].values  # Labels of training

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=123)

rf1 = RandomForestClassifier(n_estimators=300, criterion='gini', max_features="sqrt",
                             oob_score=True, n_jobs=13, max_depth=14,
                             verbose=0)

rf1.fit(x_train, y_train)

ff_his = flatforest(rf1, x_test)
nt_his = nerftab(ff_his)
t1 = localnerf(nt_his, 1)
t2 = localnerf(nt_his, 2)
t1.to_csv('testoutput_his1.txt', sep='\t')
t2.to_csv('testoutput_his0.txt', sep='\t')
