import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from statistics import mode
from sklearn.naive_bayes import GaussianNB

test = pd.read_csv('census-income.test.csv')
train = pd.read_csv('census-income.data.csv')
test = test.drop(columns=['label'])
train = train.drop(columns=['label'])

# data preprocessing

# imputing data in empty cells
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

# converting sex to binary female = 0 male = 1
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

# converting nationality to continent/region as there are too many values for one hot encoding
caribbean = [' Haiti', ' Puerto-Rico', ' Jamaica', ' Cuba', ' Dominican-Republic', ' Trinadad&Tobago']
asia = [' Japan', ' Taiwan', ' Hong', ' Philippines', ' Cambodia', ' India', ' China', ' Iran', ' Vietnam', ' Laos', ' Thailand']
south_america = [' Peru', ' Columbia', ' Ecuador']
europe = [' England', ' Germany', ' Italy', ' Poland', ' Portugal', ' Ireland', ' France', ' Hungary', ' Scotland', ' Yugoslavia', ' Holand-Netherlands', ' Greece']
central_america = [' Honduras', ' El-Salvador', ' Guatemala', ' Nicaragua']
north_america = [' United-States', ' Canada', ' Mexico']
other = [' Outlying-US(Guam-USVI-etc)', ' South']
for index in range(test.shape[0]):
    country = test.at[index, 'native-country']
    if country in caribbean:
        test.at[index, 'native-country'] = 0
    elif country in asia:
        test.at[index, 'native-country'] = 1
    elif country in south_america:
        test.at[index, 'native-country'] = 2
    elif country in europe:
        test.at[index, 'native-country'] = 3
    elif country in central_america:
        test.at[index, 'native-country'] = 4
    elif country in north_america:
        test.at[index, 'native-country'] = 5
    elif country in other:
        test.at[index, 'native-country'] = 6
for index in range(train.shape[0]):
    country = train.at[index, 'native-country']
    if country in caribbean:
        train.at[index, 'native-country'] = 0
    elif country in asia:
        train.at[index, 'native-country'] = 1
    elif country in south_america:
        train.at[index, 'native-country'] = 2
    elif country in europe:
        train.at[index, 'native-country'] = 3
    elif country in central_america:
        train.at[index, 'native-country'] = 4
    elif country in north_america:
        train.at[index, 'native-country'] = 5
    elif country in other:
        train.at[index, 'native-country'] = 6

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

# one hot encoding discrete features
race_test_features, race_train_features = one_hot_encode('race')
education_test_features, education_train_features = one_hot_encode('education')
workclass_test_features, workclass_train_features = one_hot_encode('workclass')
marital_test_features, marital_train_features = one_hot_encode('marital-status')
relationship_test_features, relationship_train_features = one_hot_encode('relationship')
occupation_test_features, occupation_train_features = one_hot_encode('occupation')
native_country_test_features, native_country_train_features = one_hot_encode('native-country')

# creating table of one hot encoded testing data along with continuous data
test_ohe = pd.concat([test['age'], test['fnlwgt'], test['education-num'], test['capital-gain'], test['capital-loss'], test['hours-per-week'], race_test_features, education_test_features, workclass_test_features, marital_test_features, relationship_test_features, occupation_test_features, native_country_test_features], axis=1)
test_ohe.head()

# creating table of one hot encoded training data along with continuous data
train_ohe = pd.concat([train['age'], train['fnlwgt'], train['education-num'], train['capital-gain'], train['capital-loss'], train['hours-per-week'], race_train_features, education_train_features, workclass_train_features, marital_train_features, relationship_train_features, occupation_train_features, native_country_train_features], axis=1)
train_ohe.head()

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
train_labels = train_labels['label']
train_labels = train_labels.astype('int')
test_labels = test_labels['label']
test_labels = test_labels.astype('int')

# using SMOTE on training data because the data is imbalanced
sm = SMOTE(random_state=2)
train_features_balanced, train_labels_balanced = sm.fit_sample(train_ohe, train_labels)
test_features = test_ohe

#data mining

#svm with rbf - really slow
svclassifier_rbf = SVC(kernel='rbf')
svclassifier_rbf.fit(train_features_balanced, train_labels_balanced)
predictor_rbf = svclassifier_rbf.predict(test_features)
print(confusion_matrix(test_labels,predictor_rbf))
print(classification_report(test_labels,predictor_rbf))

#knn
knn_classifier = KNeighborsClassifier(n_neighbors=1027)
knn_classifier.fit(train_features_balanced, train_labels_balanced) 
knn_predict = knn_classifier.predict(test_features)
print(confusion_matrix(test_labels, knn_predict))
print(classification_report(test_labels,knn_predict))

#naive bayes
gnb = GaussianNB()
gnb_pred = gnb.fit(train_features_balanced, train_labels_balanced).predict(test_features)
print(confusion_matrix(test_labels, gnb_pred))
print(classification_report(test_labels, gnb_pred))

#printing overall accuracy
final = []
for i in range(len(predictor_rbf)):
    final.append(mode([knn_predict[i], predictor_rbf[i], gnb_pred[i]]))
wrongs = 0
for i in range(len(test_labels)):
    if final[i] != test_labels[i]:
        wrongs += 1
print("accuracy:", 1-(wrongs/len(test_labels)))

