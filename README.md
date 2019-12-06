# data-mining-project
Project for data-mining.

Yijun Zhao
Data Mining
12/6/19
Made by Josh Choi, Helen Dempsey, Ben Vecchio, Claudia Westby

## Objective
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
## To run
Have the most up-to-date version of Python or Anaconda installed.
Ensure that all files are within the same folder.

### In Terminal/Command Line:

1. ```$ cd ``` to whichever directory where the files are located
2. Run command ```$ python3 income-classifier.py ```

### In Anaconda Navigator:

1. Launch Jupyter Notebook
2. Navigate to folder where files are located
3. Open final_project.ipynb
4. In the toolbar, select Cell and "Run All"

---

NOTES: 

1. The message "Using TensorFlow backend" is not an error but simply an information message.
2. Values output on your run may be different from the values recorded in our report. This is due to SMOTE's random selection.
