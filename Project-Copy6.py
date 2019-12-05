#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from statistics import mode
from sklearn.naive_bayes import GaussianNB


# In[2]:


test = pd.read_csv('census-income.test.csv')
train = pd.read_csv('census-income.data.csv')
test = test.drop(columns=['label'])
train = train.drop(columns=['label'])


# In[3]:


#data preprocessing


# In[4]:


#imputing data in empty cells
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


# In[5]:


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


# In[6]:


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


# In[7]:


race_test_features, race_train_features = one_hot_encode('race')
education_test_features, education_train_features = one_hot_encode('education')
workclass_test_features, workclass_train_features = one_hot_encode('workclass')
marital_test_features, marital_train_features = one_hot_encode('marital-status')
relationship_test_features, relationship_train_features = one_hot_encode('relationship')
occupation_test_features, occupation_train_features = one_hot_encode('occupation')
native_country_test_features, native_country_train_features = one_hot_encode('native-country')


# In[10]:


#creating table of one hot encoded testing data
#creating table of one hot encoded testing data
test_ohe = pd.concat([test['age'], test['fnlwgt'], test['education-num'], test['capital-gain'], test['capital-loss'], test['hours-per-week'], race_test_features, education_test_features, workclass_test_features, marital_test_features, relationship_test_features, occupation_test_features, native_country_test_features], axis=1)
test_ohe[' Holand-Netherlands'] = 0
test_ohe.head()


# In[13]:


#creating table of one hot encoded training data
train_ohe = pd.concat([train['age'], train['fnlwgt'], train['education-num'], train['capital-gain'], train['capital-loss'], train['hours-per-week'], race_train_features, education_train_features, workclass_train_features, marital_train_features, relationship_train_features, occupation_train_features, native_country_train_features], axis=1)
train_ohe.head()


# In[14]:


#creating array of test labels
test_labels = pd.read_csv('census-income.test.csv')
test_labels = test_labels.drop(columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'hours-per-week', 'capital-loss', 'native-country'])
#creating array of training labels
train_labels = pd.read_csv('census-income.data.csv')
train_labels = train_labels.drop(columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'hours-per-week', 'capital-loss', 'native-country'])
#converting labels into binary: 0 for less than 50K, 1 for greater than 50K
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
#converting labels from panda dataframe to flattened array
unflattened = train_labels
train_labels = train_labels['label']
train_labels = train_labels.astype('int')
test_labels = test_labels['label']
test_labels = test_labels.astype('int')


# In[15]:


#using SMOTE on training data because the data is imbalanced
sm = SMOTE(random_state=2)
train_features_balanced, train_labels_balanced = sm.fit_sample(train_ohe, train_labels)
test_features = test_ohe


# In[16]:


#data mining 


# In[ ]:


#svm with rbf 
svclassifier_rbf = SVC(kernel='rbf')
svclassifier_rbf.fit(train_features_balanced, train_labels_balanced)
predictor_rbf = svclassifier_rbf.predict(test_features)


# In[ ]:


print(confusion_matrix(test_labels,predictor_rbf))
print(classification_report(test_labels,predictor_rbf))


# In[ ]:


#svm with polynomial kernel
svclassifier_polynomial = SVC(kernel='polynomial', degree=5)
svclassifier_polynomial.fit(train_features_balanced, train_labels_balanced)
predictor_polynomial = svclassifier_polynomial.predict(test_features)


# In[ ]:


print(confusion_matrix(test_labels,predictor_polynomial))
print(classification_report(test_labels,predictor_polynomial))


# In[51]:


#knn
knn_classifier = KNeighborsClassifier(n_neighbors=1027)
knn_classifier.fit(train_features_balanced, train_labels_balanced) 
knn_predict = knn_classifier.predict(test_features)


# In[52]:


print(confusion_matrix(test_labels, knn_predict))


# In[53]:


print(classification_report(test_labels,knn_predict))


# In[17]:


#naive bayes
gnb = GaussianNB()
gnb_pred = gnb.fit(train_features_balanced, train_labels_balanced).predict(test_features)


# In[18]:


print(confusion_matrix(test_labels, gnb_pred))
print(classification_report(test_labels, gnb_pred))


# In[47]:


#printing overall accuracy
final = []
for i in range(len(predictor_rbf)):
    final.append(mode([knn_predict[i], predictor_rbf[i], predictor_polynomial[i]]))
wrongs = 0
for i in range(len(test_labels)):
    if final[i] != test_labels[i]:
        wrongs += 1
print(1-(wrongs/len(test_labels)))


# In[ ]:





# In[ ]:




