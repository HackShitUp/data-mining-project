# data-mining-project
```
Yijun Zhao
Data Mining
12/6/19
Made by Josh Choi, Helen Dempsey, Ben Vecchio, Claudia Westby
```

# About
The objective of this project is the following:
```
Prediction task is to determine whether a person makes over 50K a year. You should use census-income.data to train your classifier and use census-income.test to evaluate the performance of your learning algorithm.
```
Our algorithm manages the following inconsistencies:
```
– Handling imbalanced dataset (aka different amounts of positive and negative labels).
– Proper imputation methods for missing values.
– Different treatment of various type of features: continuous, discrete, categorical, etc.
```

# Project Architecture
```
•
└── income_classifier.py
└── data
    └── census-income.data.csv
    └── census-income.test.csv
    └── census-income.names.csv.txt
```

# Build Requirements
Latest version of ```Python```.

# Run Instructions
1. ```cd``` to the root directory .
2. Ensure all datasets are in the "data" folder.
2. Run the following command:
```
python income_classifier.py
```
