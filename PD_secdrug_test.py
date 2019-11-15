# An instance of NERF
# Using CCLE gene expression data, PD_drug test apart from Lapatinib
# NERF V0.3
# Yue Zhang <yue.zhang@lih.lu>

import mygene
import networkx as nx
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

# %%
if platform.system() == 'Windows':
    # Windows in the lab
    gdscic = pd.read_csv('P:/VM/Drug/data/output/GDSCIC50.csv')
    ccleic = pd.read_csv('P:/VM/Drug/data/output/CCLEIC50.csv')
if platform.system() == 'Darwin':
    # My mac
    gdscic = pd.read_csv('/Users/yue/Pyc/Drug2018Sep/data/GDSCIC50.csv')
    ccleic = pd.read_csv('/Users/yue/Pyc/Drug2018Sep/data/CCLEIC50.csv')

# Extract the Lat. Drug data from both of the datasets

pdccle = ccleic.loc[(ccleic.drug == 'PD0325901')]

# Create list for subset
cipd = list(range(3, len(pdccle.columns), 1))
cipd.insert(0, 1)

# subset two sets
pdC = pdccle.iloc[:, cipd]
# pdC.head(1)
# %%
# -------------------------###
# Get familiar with python DS
# ref: https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction

# Label the class
le = LabelEncoder()
le_count = 0

# iterate through columns
for col in pdC:
    if pdC.loc[:, col].dtype == 'object':
        # if less than 2 classes(which is better to use one-hot coding if not)
        if len(list(pdC.loc[:, col].unique())) <= 2:
            # 'train' the label encoder with the training data
            le.fit(pdC.loc[:, col])
            # Transform both training and testing
            pdC.loc[:, col] = le.transform(pdC.loc[:, col])
            # pdC.loc[:, col] = le.transform(pdC.loc[:, col])

            # Keep track of how many columns were labeled
            le_count += 1

print('%d columns were label encoded.' % le_count)

# %%
# Exploratory Data Analysis(EDA)

# Distribution of the target classes(columns)
pdC['SENRES'].value_counts()
pdC['SENRES'].head(4)

pdC['SENRES'].plot.hist()
plt.show()


# %%
# Examine Missing values
def missing_value_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

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

Missing_values = missing_value_table(pdC)
Missing_values.head(10)
# %%
# Column Types
# Number of each type of column
pdC.dtypes.value_counts()

# Check the number of the unique classes in each object column
pdC.select_dtypes('object').apply(pd.Series.nunique, axis=0)

# %%
# Correlations
correlations = pdC.iloc[:, 0:200].corr()['SENRES'].sort_values(na_position='first')

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))
# Create Cross-validation and training/testing


# %%
# Random forest 1st

# Define the RF
random_forest = RandomForestClassifier(n_estimators=400, random_state=123, max_features="sqrt",
                                       criterion="gini", oob_score=True, n_jobs=10, max_depth=12,
                                       verbose=0)
# %%
# Drop SENRES

train_labels = pdC.loc[:, "SENRES"]
cell_lines_pdC = pdC.loc[:, "ccle.name"]
pdC = pdC.drop(['ccle.name'], axis=1)

if 'SENRES' in pdC.columns:
    train = pdC.drop(['SENRES'], axis=1)
else:
    train = pdC.copy()
# train.iloc[0:3,0:3]
features = list(train.columns)
# train["SENRES"] = train_labels


# %%

# RF 1st train 5 trees

random_forest.fit(train, train_labels)

# Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
#
# feature_importances
# train.shape
# # Make predictions on the test data
# test_labels = lapaG.loc[:, "SENRES"]
# cell_lines_lapaG = lapaG.loc[:, "gdsc.name"]
#
# if 'SENRES' in lapaG.columns:
#     test = lapaG.drop(['SENRES'], axis=1)
# else:
#     test = lapaG.copy()
#
# test = test.drop(['gdsc.name'], axis=1)
# predictions = random_forest.predict(test)
# predictions
#
# confusion_matrix(test_labels, predictions)
# %%
random_forest.oob_score_
testC = train.iloc[[308]]

# #
# test_pred = random_forest.predict(testC)
# test_labels = pdC.loc[[2, 4]:, "SENRES"]
# random_forest.decision_path(testC)
# len(list(test_pred))
# print(confusion_matrix(test_labels, test_pred))
# random_forest.get_params(deep=True)
#


# %%
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_graphviz
# import graphviz
# tree1 = random_forest.estimators_[2]
# from sklearn import tree
# dotdata = export_graphviz(tree1, out_file=None,
#                 feature_names=train.columns,
#                 rounded=True, proportion=False,
#                 precision=2,filled=True)
#
# graph5 = graphviz.Source(dotdata)
# graph5.render("test")
# from subprocess import call
# import os
# dir_path = os.getcwd()
#
# tree1.tree_.impurity
#
# print(dotdata)
# %%
# Check cross-validation result
# from sklearn.model_selection import cross_val_score
# print(np.mean(cross_val_score(random_forest, train, train_labels, cv=10)))
# print(cross_val_score(random_forest, train, train_labels, cv=10))

# %%
# Module for the network approaches
# Extract the information in the decision tree
#
# n_nodes = random_forest.estimators_[2].tree_.node_count
# children_left = random_forest.estimators_[2].tree_.children_left
# children_right = random_forest.estimators_[2].tree_.children_right
# feature = random_forest.estimators_[2].tree_.feature
# threshold = random_forest.estimators_[2].tree_.threshold
# testy = train.iloc[[2, 4]]
# decision_p = random_forest.decision_path(testy)
# leave_p = random_forest.apply(test)
# decision_p[0].indices
# testy.shape
# print(decision_p)


# %%
# Create feature list, convert ENSG into gene symbols
featurelist = train.columns.values.tolist()
# Mygene convertion
mg = mygene.MyGeneInfo()
mg.metadata('available_fields')
con = mg.querymany(featurelist, scopes='ensembl.gene', fields='symbol', species="human", as_dataframe=True)
# replace Nan unmapped with original ENSGZ
con['symbol'] = np.where(con['notfound'] == True, con.index.values, con['symbol'])

featurelist_g = con.iloc[:, 3].reset_index()
feag = featurelist_g.iloc[:, 1]
# featurelist_g.loc[featurelist_g['query'] == 'ENSG00000229425'].index[0]


feag.pop(47081)

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

pd_ff.loc[pd_ff["node_type"] == "decision_node", :].shape
# %%
# Other cancer
# Uninarytract bladder 2, A549 lung 16, BT549 Breast 31
pd_ff = flatforest(random_forest, testC)
pd_f = extarget(random_forest, testC, pd_ff)
pd_nt = nerftab(pd_f)
s3 = localnerf(pd_nt, 0)
s5 = localnerf(pd_nt, 1)
H322 = localnerf(pd_nt, 0)
r3 = twonets(s3, "sample 3 in training", index, feag)
r5 = twonets(s5, "sample 5 in training", index, feag)
r322 = twonets(H322, "H322 in training", index, feag)

# %%
# Feature importance list

feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': featurelist, 'importance': feature_importance_values})
feature_importances.to_csv(os.getcwd() + '/output/featureimpPD_feb.txt', sep='\t')

# #############################################################################
# Classification and ROC analysis
# %%
X = train
y = train_labels
X = X.values
y = y.values

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=6)
classifier = random_forest

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic CV')
plt.legend(loc="lower right")
plt.savefig('Per.png')
plt.show()

# TODO put the re-naming of the features after the extraction

# %%

import matplotlib.pyplot as plt

correlations = pdC.iloc[:, 1:200].corr()
plt.matshow(pdC.iloc[:, 1:len(pdC.columns)].corr())
plt.show()

sol = (
    correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))

# %% Correlation matrix adpot from kaggle
# TODO Need some modification
correlations = train_df[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.head(10)
correlations.tail(10)
