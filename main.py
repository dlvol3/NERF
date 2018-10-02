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
random_forest = RandomForestClassifier(n_estimators=200, random_state=123, max_features="sqrt",
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
visualize_classifier(random_forest, train, test)



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
testy = train.iloc[[2,4]]
decision_p = random_forest.decision_path(testy)
leave_p = random_forest.apply(test)
decision_p[0].indices
testy.shape
print(decision_p)
#%%
print(tree1.tree_.children_left)


#%%
for rf_tree in random_forest.estimator:
    n_nodes = random_forest.estimators_[1].tree_.node_count
    children_left = random_forest.estimators_[1].tree_.children_left
#%%
# Tree pre-order traversal
for i in range(tree1.tree_.node_count):
    print("%s, %s, %s"%(i, tree1.tree_.children_left[i], tree1.tree_.children_right[i]))
#%%
print(decision_p[1])

#%%
# Generating the allinone table


def flatforest(rf, testdf):
    from time import time
    tt = time()
    tree_infotable = pd.DataFrame()
    raw_hits = pd.DataFrame()

    for t in range(rf.n_estimators):
        # Generate the info table for trees

        # Preparation

        # Node index # Count from leftleft first /list
        nodeIndex = list(range(0, rf.estimators_[t].tree_.node_count, 1))
        # Node index forest level
        nodeInForest = list(map(lambda x: x + rf.decision_path(testdf)[1].item(t), nodeIndex))
        # lc # left children of each node, by index ^ /ndarray 1D
        lc = rf.estimators_[t].tree_.children_left
        # rc # right children of each node, by index ^ /ndarray 1D
        rc = rf.estimators_[t].tree_.children_right
        # Proportion of sample in each nodes  /ndarray +2D add later
        # TODO if the pv info is needed for the weighted GS score, re-calculate. No need to add it into the table.
        pv = rf.estimators_[t].tree_.value
        # Feature index, by index /1d array
        featureIndex = rf.estimators_[t].tree_.feature
        # Feature threshold, <= %d %threshold
        featureThreshold = rf.estimators_[t].tree_.threshold
        # Gini impurity of the node, by index
        gini = rf.estimators_[t].tree_.impurity
        # Tree index
        treeIndex = t+1
        testlist = pd.DataFrame(
            {'node_index': nodeIndex,
             'left_c': lc,
             'right_c': rc,
             'feature_index': featureIndex,
             'feature_threshold': featureThreshold,
             'gini': gini,
             'tree_index': treeIndex,
             'nodeInForest': nodeInForest
             })

        # Calculation of the default gini gain
        gslist = list()
        nodetype = list()
        for ii in range(rf.estimators_[t].tree_.node_count):
            if testlist.loc[:, 'feature_index'][ii] == -2:
                gslist.append(-1)
                nodetype.append("leaf_node")
                continue  # Next if node is leaf

            ri = testlist.loc[:, 'right_c'][ii]  # right child index of node i
            li = testlist.loc[:, 'left_c'][ii]  # left child index of node i

            gs_index = testlist.loc[:, 'gini'][ii] \
                - np.sum(pv[li])/np.sum(pv[ii])*testlist.loc[:, 'gini'][li] \
                - np.sum(pv[ri])/np.sum(pv[ii])*testlist.loc[:, 'gini'][ri]

            gslist.append(gs_index)
            nodetype.append("decision_node")

        testlist['GS'] = pd.Series(gslist).values
        testlist['node_type'] = pd.Series(nodetype).values

        tree_infotable = pd.concat([tree_infotable, testlist])
    print("Forest %s flatted, matrix generate with %d rows and %d columns" % (rf, tree_infotable.shape[0],
                                                                             tree_infotable.shape[1]))
    print("Run time for extracting tree information:")
    print(time() - tt)

    tt2 = time()

    for s_index in range(rf.decision_path(testdf)[0].indptr.shape[0] - 1):  # Loop on samples for prediction
        sample_ceiling = rf.decision_path(testdf)[0].indptr[s_index + 1]  # The ceiling hit index of the current sample
        sample_floor = rf.decision_path(testdf)[0].indptr[s_index]
        hitall = pd.DataFrame()
        predictlist = list()   # Store the predictions among the forest for a certain sample
        treelist = list()
        samplelist = s_index
        predictlist_for_all = pd.DataFrame()
        for ttt in range(rf.n_estimators):
            pred_s_t = rf.estimators_[ttt].predict(testy)[s_index]
            predictlist.append(pred_s_t)
            treelist.append(ttt)
        predictlist_for_sample = pd.DataFrame(
            {'prediction': predictlist,
             'tree index': treelist,
             'sample': samplelist
             })
        predictlist_for_sample['matching'] = np.where(predictlist_for_sample['prediction'] ==
                                                   random_forest.predict(testy)[predictlist_for_sample['sample']], 'match', 'not_matching')
        predictlist_for_all = pd.concat([predictlist_for_all, predictlist_for_sample])

        for hit_index in range(sample_floor, sample_ceiling):  # Loop through the hits of the current sample
            hit = tree_infotable.loc[tree_infotable['nodeInForest'] == rf.decision_path(testdf)[0].indices[hit_index],
                        ['feature_index', 'GS', 'tree_index','feature_threshold']]
            hit['sample_index'] = pd.Series(s_index).values
            hitall = pd.concat([hitall, hit])
        raw_hits = pd.concat([raw_hits, hitall])

    df = list()
    df.extend((tree_infotable, raw_hits, predictlist_for_all))
    print("All node used for predicting samples extracted")
    print("Run time for generate the decision table:")
    print(time() - tt2)
    print("Total run time:")
    print(time() - tt)
    return df
#%%
# random_forest.predict(testy)[1]

TIE = flatforest(random_forest, testy)


#%%
testy.shape
random_forest.decision_path(testy)[0].indices
#%%
# All possible pairs generator
# TODO get rid of the leaves
# TODO give the tree a index?
# TODO Same to previous one, give sample a index to loop with
raw_hits
max(raw_hits.loc[:, 'sample_index'])  # Number of samples
for s_in in range(max(raw_hits.loc[:, 'sample_index'])):
    single_tree = df[1].loc[df[1]['sample_index' == s_in],]#  TODO columns add or not?
    for t_in in range(max(single_tree))


max(raw_hits.loc[:, 'tree_index'])  # 200 trees in the test case
for n in range(max(raw_hits.loc[:, 'tree_index'])):


