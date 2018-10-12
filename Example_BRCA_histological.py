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
import mygene
# NewtworkX
import math
import networkx as nx


#

if platform.system() == 'Windows':
    # Windows in the lab
    B20000 = pd.read_table("P:/VM/TCGA/Data/BRCA/BMReady.txt", sep='\t')
if platform.system() == 'Darwin':
    # My mac
    B20000 = pd.read_table("/Users/yue/Pyc/NERF-RF_interpreter/data/BMReady.txt", sep='\t')
# Read in data and display first 5 rows

# Creating the dependent variable class
factor = pd.factorize(B20000['histological_type'])
B20000.histological_type = factor[0]
definitions = factor[1]

x = B20000.iloc[:, 1:20530].values   # Features for training
y = B20000.iloc[:, 0].values  # Labels of training

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=123)

rf1 = RandomForestClassifier(n_estimators=30, criterion='gini', max_features="sqrt",
                             oob_score=True, n_jobs=13, max_depth=14,
                             verbose=0)

rf1.fit(x_train, y_train)

ff_his = flatforest(rf1, x_test)
nt_his = nerftab(ff_his)
t1 = localnerf(nt_his, 1)
t2 = localnerf(nt_his, 2)
t1.to_csv('testoutput_his1.txt', sep='\t')
t2.to_csv('testoutput_his0.txt', sep='\t')


# Lap example
#%%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification

import platform
#%%
if platform.system() == 'Windows':
    # Windows in the lab
    gdscic = pd.read_csv('P:/VM/Drug/data/output/GDSCIC50.csv')
    ccleic = pd.read_csv('P:/VM/Drug/data/output/CCLEIC50.csv')
if platform.system() == 'Darwin':
    # My mac
    gdscic = pd.read_csv('/Users/yue/Pyc/Drug2018Sep/data/GDSCIC50.csv')
    ccleic = pd.read_csv('/Users/yue/Pyc/Drug2018Sep/data/CCLEIC50.csv')

# Extract the Lat. Drug data from both of the datasets
gdscic.head(5)
lapagdsc = gdscic.loc[(gdscic.drug == 'Lapatinib')]
lapaccle = ccleic.loc[(ccleic.drug == 'Lapatinib')]

# Create list for subset
ciLapa = list(range(3, len(lapaccle.columns), 1))
ciLapa.insert(0, 1)

# subset two sets
lapaC = lapaccle.iloc[:, ciLapa]
lapaG = lapagdsc.iloc[:, ciLapa]
lapaC.head(1)
#%%
# -------------------------###
# Get familiar with python DS
# ref: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

# Label the class
le = LabelEncoder()
le_count = 0

# iterate through columns
for col in lapaC:
    if lapaC.loc[:, col].dtype == 'object':
        # if less than 2 classes(which is better to use one-hot coding if not)
        if len(list(lapaC.loc[:, col].unique())) <= 2:
            # 'train' the label encoder with the training data
            le.fit(lapaC.loc[:, col])
            # Transform both training and testing
            lapaC.loc[:, col] = le.transform(lapaC.loc[:, col])
            lapaG.loc[:, col] = le.transform(lapaG.loc[:, col])

            # Keep track of how many columns were labeled
            le_count += 1

print('%d columns were label encoded.' % le_count)

#%%
# Exploratory Data Analysis(EDA)

# Distribution of the target classes(columns)
lapaC['SENRES'].value_counts()
lapaC['SENRES'].head(4)

lapaC['SENRES'].plot.hist()
plt.show()

#%%
# Examine Missing values
def missing_value_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum()/len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing values', 1: '% of Total Values'}
    )

    # Sort the table by percentage of the missing values
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print the summary
    print("Your selected data frame has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the result
    return mis_val_table_ren_columns


# Check the missing value in the dataset

Missing_values = missing_value_table(lapaC)
Missing_values.head(10)
#%%
# Column Types
# Number of each type of column
lapaC.dtypes.value_counts()

# Check the number of the unique classes in each object column
lapaC.select_dtypes('object').apply(pd.Series.nunique, axis=0)

#%%
# Correlations
correlations = lapaC.iloc[:, 0:200].corr()['SENRES'].sort_values(na_position='first')

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))
# Create Cross-validation and training/testing


#%%
# Random forest 1st

# Define the RF
random_forest = RandomForestClassifier(n_estimators=300, random_state=123, max_features="sqrt",
                                       criterion="gini", oob_score=True, n_jobs=10, max_depth=12,
                                       verbose=0)
#%%
# Drop SENRES

train_labels = lapaC.loc[:, "SENRES"]
cell_lines_lapaC = lapaC.loc[:, "ccle.name"]
lapaC = lapaC.drop(['ccle.name'], axis=1)

if 'SENRES' in lapaC.columns:
    train = lapaC.drop(['SENRES'], axis=1)
else:
    train = lapaC.copy()
train.iloc[0:3,0:3]
features = list(train.columns)
# train["SENRES"] = train_labels



#%%

# RF 1st train 5 trees

random_forest.fit(train, train_labels)

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
feature_importances
train.shape
# Make predictions on the test data
test_labels = lapaG.loc[:, "SENRES"]
cell_lines_lapaG = lapaG.loc[:, "gdsc.name"]

if 'SENRES' in lapaG.columns:
    test = lapaG.drop(['SENRES'], axis=1)
else:
    test = lapaG.copy()

test = test.drop(['gdsc.name'], axis=1)
predictions = random_forest.predict(test)
predictions

confusion_matrix(test_labels, predictions)
#%%
random_forest.oob_score_
#
test_pred = random_forest.predict(test)
random_forest.decision_path(test)
len(list(test_pred))
print(confusion_matrix(test_labels, test_pred))
random_forest.get_params(deep=True)




#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
tree1 = random_forest.estimators_[2]
from sklearn import tree
dotdata = export_graphviz(tree1, out_file=None,
                feature_names=train.columns,
                rounded=True, proportion=False,
                precision=2,filled=True)

graph5 = graphviz.Source(dotdata)
graph5.render("test")
from subprocess import call
import os
dir_path = os.getcwd()

tree1.tree_.impurity

print(dotdata)
#%%
# Check cross-validation result
# from sklearn.model_selection import cross_val_score

# print(np.mean(cross_val_score(random_forest, train, train_labels, cv=10)))
# print(cross_val_score(random_forest, train, train_labels, cv=10))

#%%
# Module for the network approaches
# Extract the information in the decision tree

n_nodes = random_forest.estimators_[2].tree_.node_count
children_left = random_forest.estimators_[2].tree_.children_left
children_right = random_forest.estimators_[2].tree_.children_right
feature = random_forest.estimators_[2].tree_.feature
threshold = random_forest.estimators_[2].tree_.threshold
testy = train.iloc[[2, 4]]
decision_p = random_forest.decision_path(testy)
leave_p = random_forest.apply(test)
decision_p[0].indices
testy.shape
print(decision_p)


#%%
# Create feature list, convert ENSG into gene symbols
featurelist = train.columns.values.tolist()
# Mygene convertion
mg = mygene.MyGeneInfo()
mg.metadata('available_fields')
con = mg.querymany(featurelist, scopes='ensembl.gene', fields='symbol', species="human", as_dataframe=True)
# replace Nan unmapped with original ENSG
con['symbol'] = np.where(con['notfound'] == True, con.index.values, con['symbol'])

featurelist_g = con.iloc[:, 3].reset_index()
feag = featurelist_g.iloc[:, 1]
feag.pop(7220)
feag.pop(38223)
feag.pop(47083)

# POP out those duplicates
feag = list(feag)

index = list(range(len(featurelist)))

sl = random_forest.feature_importances_
fl = pd.DataFrame({
    'feature_name': feag,
    'score': sl,
    'index': index
})


fls = fl.sort_values('score', ascending=False)

#%%


# Sort by value, Dict
def sort_by_value(d):
    items = d.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)
    return [backitems[i][1] for i in range(0, len(backitems))]
#%%


#%%
# Other cancer
# Uninarytract bladder 2, A549 lung 16, BT549 Breast 31
testy2 = train.iloc[[2, 16, 31]]
TIE2 = flatforest(random_forest, testy2)
nt_lap2 = nerftab(TIE2)
tbla = localnerf(nt_lap2, 0)
tlung = localnerf(nt_lap2, 1)
tbreast = localnerf(nt_lap2, 2)

#%%
#Define a func for later


def twonets(outdf, filename, index1=2, index2=10):
    """
    Purpose
    ---------
    Process the result of the localnerf(), return two networks, one with everything, one with less info
    ---------
    :param outdf: The localnerf() result
    :param filename: the desire filename, or path with name, no suffix
    :param index1: Index for selecting top degree of centrality, default = 2
    :param index2: Index for selecting top edge intensity, default = 10
    :return: A list contains five elements, whole network with gene names, degree of all features,
     degreetop selected, eitop delected, sub network with gene names
    """
    outdf = outdf.replace(index, feag)
    # export the 'everything' network
    outdf.to_csv(filename + "_everything.txt", sep='\t')

    gout = nx.from_pandas_edgelist(outdf, "feature_i", "feature_j", "EI")

    degreecout = nx.degree_centrality(gout)
    # Test save the centrality
    np.save(filename + "_DC.txt", degreecout)
    # Large to small sorting
    sortdegree = sort_by_value(degreecout)
    # take the top sub of the DC
    degreetop = sortdegree[: int(index1 * math.sqrt(len(sortdegree)))]
    # Large to small sorting, Edge intensity
    outdfsort = outdf.sort_values('EI', ascending=False)

    eitop = outdfsort[: int(index2 * math.sqrt(outdfsort.shape[0]))]

    outdffinal = eitop[eitop['feature_i'].isin(degreetop) & eitop['feature_j'].isin(degreetop)]
    outdffinal.to_csv(filename + '_sub.txt', sep='\t')
    outputfunc = list()
    outputfunc.extend((outdf, degreecout, degreetop, eitop, outdffinal))
    return outputfunc

#%%


BT549 = twonets(tbreast, "BT549_breast")
UT = twonets(tbla, "UT_bladder")
A549 = twonets(tlung, "A549_lung")


#%%
# Similarity between the two predictions


