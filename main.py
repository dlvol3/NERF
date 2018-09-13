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


#%%
gdscic = pd.read_csv('P:/VM/Drug/data/output/GDSCIC50.csv')
ccleic = pd.read_csv('P:/VM/Drug/data/output/CCLEIC50.csv')

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
random_forest = RandomForestClassifier(n_estimators=1000, random_state=123, max_features="sqrt",
                                       criterion="gini", oob_score=True, n_jobs=10, max_depth=12,
                                       verbose=0, class_weight="balanced")
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
train["SENRES"] = train_labels

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
predictions = random_forest.predict_proba(test)[:, 1]
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
tree1 = random_forest.estimators_[5]
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

