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
# random_forest.predict(testy)[1]

TIE = flatforest(random_forest, testy)
nt_lap = nerftab(TIE)
t1 = localnerf(nt_lap, 1)
t0 = localnerf(nt_lap, 0)
t1.to_csv('testoutput_lap1.txt', sep='\t')
t0.to_csv('testoutput_lap0.txt', sep='\t')

#%%

featurelist = train.columns.values.tolist()
# Mygene convertion
mg = mygene.MyGeneInfo()
mg.metadata('available_fields')
con = mg.querymany(featurelist, scopes='ensembl.gene', fields='symbol', species="human", as_dataframe=True)
featurelist_g = con.iloc[:, 3].reset_index()
feag = featurelist_g.iloc[:, 1]
feag.pop(7220)
feag.pop(38223)
feag.pop(47083)
feag = list(feag)
con.shape
featurelist[0]   # 100+ in original FI, top 5 in NERF
featurelist.index('ENSG00000236107')
featurelist.index('ENSG00000229425')
featurelist.index('ENSG00000233024')

index = list(range(len(featurelist)))

sl = random_forest.feature_importances_
fl = pd.DataFrame({
    'feature_name': featurelist,
    'score': sl,
    'index': index
})


fls = fl.sort_values('score', ascending=False)

fls.loc[fls['feature_name'] == 'ENSG00000189077', ]
#%%


# Sort by value, Dict
def sort_by_value(d):
    items = d.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort(reverse=True)
    return [backitems[i][1] for i in range(0, len(backitems))]
#%%
# NewtworkX
import math
import networkx as nx

t0RT = t0.replace(index, feag)

G = nx.from_pandas_edgelist(t0RT, "feature_i", "feature_j", "EI")

DC = nx.degree_centrality(G)
RTDC = sort_by_value(DC)
RTDC1 = RTDC[: int(2*math.sqrt(len(RTDC)))]

t0RT = t0RT.sort_values('EI', ascending=False)
t01 = t0RT[: int(10*math.sqrt(t0RT.shape[0]))]


t02 = t01[t01['feature_i'].isin(RTDC1) & t01['feature_j'].isin(RTDC1)]
t02.to_csv('testoutput_uni.txt', sep='\t')

G = nx.from_pandas_edgelist(t02, "feature_i", "feature_j", "EI")
nx.draw(G)
plt.show()

#%%
# Other cancer
# A549 lung 16, BT549 Breast 31
testy2 = train.iloc[[16, 31]]
TIE2 = flatforest(random_forest, testy)
nt_lap2 = nerftab(TIE)
tlung = localnerf(nt_lap2, 0)
tbreast = localnerf(nt_lap2, 1)
tlung = tlung.replace(index, feag)

tbreast = tbreast.replace(index, feag)
tlung.to_csv('testoutput_A549.txt', sep='\t')
tbreast.to_csv('testoutput_BT549.txt', sep='\t')

featurelist[31107]

tlung = tlung.replace(index, feag)

tbreast = tbreast.replace(index, feag)
# replace with gene name and selection of 'powerful nodes' lung
Glung = nx.from_pandas_edgelist(tlung, "feature_i", "feature_j", "EI")

DClung = nx.degree_centrality(Glung)
RTDClung = sort_by_value(DClung)
RTDC1lung = RTDClung[: int(2*math.sqrt(len(RTDClung)))]

tlungsort = tlung.sort_values('EI', ascending=False)
tlungS = tlungsort[: int(10*math.sqrt(t0RT.shape[0]))]


tlungfinal = tlungS[tlungS['feature_i'].isin(RTDC1lung) & tlungS['feature_j'].isin(RTDC1lung)]
tlungfinal.to_csv('testoutput_RTDC1lung.txt', sep='\t')

# replace with gene name and selection of 'powerful nodes' breast
Gbreast = nx.from_pandas_edgelist(tbreast, "feature_i", "feature_j", "EI")

DCbreast = nx.degree_centrality(Gbreast)
RTDCbreast = sort_by_value(DCbreast)
RTDC1breast = RTDCbreast[: int(2*math.sqrt(len(RTDCbreast)))]

tbreastsort = tbreast.sort_values('EI', ascending=False)
tbreastS = tbreastsort[: int(10*math.sqrt(t0RT.shape[0]))]


tbreastfinal = tbreastS[tbreastS['feature_i'].isin(RTDC1breast) & tbreastS['feature_j'].isin(RTDC1breast)]
tbreastfinal.to_csv('testoutput_RTDC1breast.txt', sep='\t')

#%%
# Similarity between the two predictions
