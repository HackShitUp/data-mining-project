#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from statistics import mode
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


test = pd.read_csv('census-income.test.csv')
train = pd.read_csv('census-income.data.csv')
test = test.drop(columns=['label'])
train = train.drop(columns=['label'])


# In[ ]:


#data preprocessing


# In[ ]:


# imputing data in empty cells
# since all missing data were in discrete features, the mode was used
test_modes = test.mode()
for index in range(test.shape[0]):
    for col in test.columns:
        if test.at[index, col] == ' ?':
            test.at[index, col] = test_modes.at[0, col]
train_modes = train.mode()
for index in range(train.shape[0]):
    for col in train.columns:
        if train.at[index, col] == ' ?':
            train.at[index, col] = train_modes.at[0, col]


# In[ ]:


#converting sex to binary female = 0 male = 1
for index in range(test.shape[0]):
    if test.at[index, 'sex'] == ' Male':
        test.at[index, 'sex'] = 1
    else: 
        test.at[index, 'sex'] = 0        
for index in range(train.shape[0]):
    if train.at[index, 'sex'] == ' Male':
        train.at[index, 'sex'] = 1
    else:
        train.at[index, 'sex'] = 0


# In[ ]:


def one_hot_encode(feature):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder()
    feature_label = feature + '_label'
    #test
    feature_labels = label_encoder.fit_transform(test[feature])
    test[feature_label] = feature_labels
    test_feature_arr = onehot_encoder.fit_transform(test[[feature_label]]).toarray()
    feature_new_labels = list(label_encoder.classes_)
    feature_test_features = pd.DataFrame(test_feature_arr, columns=feature_new_labels)
    #train
    feature_labels = label_encoder.fit_transform(train[feature])
    train[feature_label] = feature_labels
    train_feature_arr = onehot_encoder.fit_transform(train[[feature_label]]).toarray()
    feature_new_labels = list(label_encoder.classes_)
    feature_train_features = pd.DataFrame(train_feature_arr, columns=feature_new_labels)
    return [feature_test_features, feature_train_features]


# In[ ]:


# one hot encoding discrete features using sklearn module
race_test_features, race_train_features = one_hot_encode('race')
education_test_features, education_train_features = one_hot_encode('education')
workclass_test_features, workclass_train_features = one_hot_encode('workclass')
marital_test_features, marital_train_features = one_hot_encode('marital-status')
relationship_test_features, relationship_train_features = one_hot_encode('relationship')
occupation_test_features, occupation_train_features = one_hot_encode('occupation')
native_country_test_features, native_country_train_features = one_hot_encode('native-country')


# In[ ]:


# creating table of one hot encoded testing data
test_ohe = pd.concat([test['age'], test['fnlwgt'], test['education-num'], test['capital-gain'], test['capital-loss'], test['hours-per-week'], race_test_features, education_test_features, workclass_test_features, marital_test_features, relationship_test_features, occupation_test_features, native_country_test_features], axis=1)
# need to add a column of ' Holand-Netherlands' since test data doesn't have a point with ' Holand-Netherlands' and training data does
test_ohe[' Holand-Netherlands'] = 0
test_ohe.head()


# In[ ]:


# creating table of one hot encoded training data
train_ohe = pd.concat([train['age'], train['fnlwgt'], train['education-num'], train['capital-gain'], train['capital-loss'], train['hours-per-week'], race_train_features, education_train_features, workclass_train_features, marital_train_features, relationship_train_features, occupation_train_features, native_country_train_features], axis=1)
train_ohe.head()


# In[ ]:


# creating array of test labels
test_labels = pd.read_csv('census-income.test.csv')
test_labels = test_labels.drop(columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'hours-per-week', 'capital-loss', 'native-country'])
# creating array of training labels
train_labels = pd.read_csv('census-income.data.csv')
train_labels = train_labels.drop(columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'hours-per-week', 'capital-loss', 'native-country'])
# converting labels into binary: 0 for less than 50K, 1 for greater than 50K
for index in range(train_labels.shape[0]):
    if train_labels.at[index, 'label'] == ' <=50K':
        train_labels.at[index, 'label'] = 0
    else:
        train_labels.at[index, 'label'] = 1
for index in range(test_labels.shape[0]):
    if test_labels.at[index, 'label'] == ' <=50K.':
        test_labels.at[index, 'label'] = 0
    else:
        test_labels.at[index, 'label'] = 1
# converting labels from panda dataframe to flattened array
unflattened = train_labels
train_labels = train_labels['label']
train_labels = train_labels.astype('int')
test_labels = test_labels['label']
test_labels = test_labels.astype('int')


# In[ ]:


# using SMOTE on training data because the data is imbalanced
sm = SMOTE()
train_features_balanced, train_labels_balanced = sm.fit_sample(train_ohe, train_labels)
test_features = test_ohe


# In[ ]:


# data mining 


# In[55]:


# decision tree
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(train_features_balanced, train_labels_balanced)
dt_predict = dt_classifier.predict(test_features)


# In[56]:


print("Decision tree confusion matrix and classification report:")
print(confusion_matrix(test_labels, dt_predict))
print(classification_report(test_labels, dt_predict))


# In[ ]:


#knn
knn_classifier = KNeighborsClassifier(n_neighbors=1027)
knn_classifier.fit(train_features_balanced, train_labels_balanced) 
knn_predict = knn_classifier.predict(test_features)


# In[ ]:


print("KNN confusion matrix and classification report:")
print(confusion_matrix(test_labels, knn_predict))
print(classification_report(test_labels,knn_predict))


# In[ ]:


#naive bayes
gnb_classifier = GaussianNB()
gnb_classifier.fit(train_features_balanced, train_labels_balanced)
gnb_predict = gnb_classifier.predict(test_features)


# In[ ]:


print("Naive bayes confusion matrix and classification report:")
print(confusion_matrix(test_labels, gnb_predict))
print(classification_report(test_labels, gnb_predict))


# In[ ]:


#printing overall accuracy
final = []
for i in range(len(dt_predict)):
    final.append(mode([knn_predict[i], dt_predict[i], gnb_pred[i]]))
wrongs = 0
for i in range(len(test_labels)):
    if final[i] == 0 and test_labels[i] == 0:
        true_negatives += 1
    elif final[i] == 0 and test_labels[i] == 1:
        false_negatives += 1
        wrongs += 1
    elif final[i] == 1 and test_labels[i] == 1:
        true_positives += 1
    elif final[i] == 1 and test_labels[i] == 0:
        false_positives += 1
        wrongs += 1


# In[ ]:


print('True Negatives (<= 50K):', true_negatives)
print('False Negatives (<= 50K):', false_negatives)
print('True Positives (>50K):', true_positives)
print('False Positives (>50K):', false_positives)


# In[ ]:


print('Final accuracy:', 1-(wrongs/len(test_labels)))

