# An instance of NERF
# Using TCGA breast cancer dataset, mRNA=>histological classes




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


