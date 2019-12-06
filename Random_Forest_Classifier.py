#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Inhibit warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[11]:


'''
Iterates through the data frame and log each value. Useful for read ops.
@param data_frame: A DataFrame object.
'''
def iterate_through_records(data_frame):
    for i, name in enumerate(data_frame.columns):
        for j, row in data_frame.iterrows():
            print(row[j])

            
            
'''
Abstract: Custom class used to impute missing values with the mean of its axis (aka column of value). 
Credits: https://stackoverflow.com/a/25562948
'''
class DataFrameImputer(TransformerMixin):

    # MARK: - Init
    def __init__(self):
        """
        Impute missing values. Columns of dtype object are imputed with the most frequent value 
        in column. Columns of other types are imputed with mean of column.
        """
        
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('?') else X[c].mean() for c in X],
            index = X.columns)
        
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# In[12]:


# Initialized array of String values representing the headers (best practice is to NOT manipualte data directly — via 'census-income.names.csv.txt')
features = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]

# Define the classifiers for $50K <= and $50K+
at_most_50K = '<=50K'
more_than_50K = '>50K'

# MARK: - DataFrame
train_data = pd.read_csv('../data/census-income.data.csv', header=None, names=features)


# In[13]:


'''
Data Preprocessing Logic:
(1) Lambda Transformation (cast String to binary - float or Int)
(2) Inpute missing values with mean of its given axis (aka take sum/count in its column)
'''

#Transform strings into integers for 'sex' column
train_data["gender"] = train_data["sex"].apply(lambda row: 0 if row == "Male" else 1)

'''
Mapping Strings to numbers is only applicable to coninuous numbers. For instance, it wouldn't make much sense to map 
'United States' to 0 'Germany' to 1, and 'Mexico' to 2. If we did this, we're saying that Mexico is more similar to 
Germany than the U.S.
'''
# Transform strings into integers for 'native-country' column
train_data["america"] = train_data["native-country"].apply(lambda row: 0 if row == "United-States" else 1)

# Create the labels
labels = train_data["income"]
# Define the key features in the data set
key_features = ["age", "capital-gain", "capital-loss", "hours-per-week", "gender", "america"]

# Split the training data and labels into training and test sets
training_data, testing_data, training_labels, testing_labels = train_test_split(train_data[key_features], labels, random_state=1)


# In[14]:


# MARK: - RandomForestClassifier
random_forest_classifier = RandomForestClassifier(random_state=1)
random_forest_classifier.fit(training_data, training_labels)

print('Important Features:', random_forest_classifier.feature_importances_)

# MARK: - Matplotlib
plt.figure(figsize=(15, 10))
plt.bar(key_features, random_forest_classifier.feature_importances_)
plt.title("Random Forest — Important Features")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.show()


# In[15]:


# Here, we define the prediction score
score = random_forest_classifier.score(testing_data, testing_labels)
score = score * 100.00
print('\nScore:')
print(score, '%')


# In[16]:


# Get the prediction from the test data
prediction_score = random_forest_classifier.predict(testing_data)
print('Prediction Score:', prediction_score)

