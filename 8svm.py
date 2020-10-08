'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), Ming Chi University of Technology (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), National Taipei University of Business (國立臺北商業大學); the Chinese Academy of R Software (中華R軟體學會); and the Data Science & Business Applications Association of Taiwan (臺灣資料科學與商業應用協會)
Notes: This code is provided without warranty.
'''

### performing OCR with SVMs
## ------------------------------------------------------------------------
import pandas as pd
letters = pd.read_csv("./_data/letterdata.csv")
# data types
print(letters.dtypes)

# Each integer value variable is between 0 and 15 (4 bits pixel value)
print(letters.describe(include = 'all'))

# The target variables are evenly distributed in each category (the default is sorted by the descending power of each frequency)
print(letters['letter'].value_counts())

from sklearn.feature_selection import VarianceThreshold
# Model definition, adaptation and conversion (i.e. delete zero-variation attributes)
vt = VarianceThreshold(threshold=0)
# no zero-variation attributes
print(vt.fit_transform(letters.iloc[:,1:]).shape)

print(np.sum(vt.get_support() == False)) # Get a mask of the features selected
# vt.get_support(indices=True) # Get integer index of the features selected

# Calculate the correlation coefficient square matrix to numpy ndarray
cor = letters.iloc[:,1:].corr().values
print(cor[:5,:5])

# correlation coefficient over (+-0.8) square matrix
import numpy as np
np.fill_diagonal(cor, 0) # Change the diagonal element value to 0
threTF = abs(cor) > 0.8
print(threTF[:5,:5])

# like R which(True and false value matrix, arr.ind=TRUE)
print(np.argwhere(threTF == True))

# Check the variable name, note that the first variable 'letter' has been excluded when calculating the correlation coefficient
print(letters.columns[1:5])

# pandas boxplot()
ax1 = letters[['xbox', 'letter']].boxplot(by = 'letter')
fig1 = ax1.get_figure()
# fig1.savefig('./_img/xbox_boxplot.png')
ax2 = letters[['ybar', 'letter']].boxplot(by = 'letter')
fig2 = ax2.get_figure()
# fig2.savefig('./_img/ybar_boxplot.png')

# train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
letters.iloc[:, 1:], letters['letter'], test_size=0.2,
random_state=0)
# data standard
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# Calculate mu and sigma of each variable of X_train
sc.fit(X_train)
# transform
X_train_std = sc.transform(X_train)
# Convert X_test with mu and sigma of each variable of X_train
X_test_std = sc.transform(X_test)

# SVC: Support Vector Classification
# SVR: Support Vector Regression
# OneClassSVM: Outlier Detection
from sklearn.svm import SVC
# Model definition (without changing the default settings), fit and conversion
svm = SVC(kernel='linear')
svm.fit(X_train_std, y_train)
tr_pred = svm.predict(X_train_std)
y_pred = svm.predict(X_test_std)
# The first 5 predictions of the training set
print(tr_pred[:5])

# The first 5 actual values of the training set
print(y_train[:5])

# The first 5 predictions of the testing set
print(y_pred[:5])

# The first 5 actual of the testing set
print(y_test[:5].tolist())

# Note that Python has another output formatting syntax (%)
err_tr = (y_train != tr_pred).sum()/len(y_train)
print("train set error: %f" % err_tr)

# The test set error rate is slightly higher than the training set error rate
err = (y_test != y_pred).sum()/len(y_test)
print("test set error: %f" % err)

# change rbf gamma to 0.2
svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
tr_pred = svm.predict(X_train_std)
y_pred = svm.predict(X_test_std)
# The first 5 predictions of the training set
print(tr_pred[:5])

# The first 5 actual values of the training set
print(y_train[:5])

# The first 5 predictions of the testing set
print(y_pred[:5])

# The first 5 actual of the testing set
print(y_test[:5].tolist())

err_tr = (y_train.values != tr_pred).sum()/len(y_train)
print("train set error: %f" % err_tr)

# The test set error rate is slightly higher than the training set error rate
err = (y_test != y_pred).sum()/len(y_test)
print("test set error: %f" % err)

# pandas_ml need：

# pip install scikit-learn==0.21.1
# pip install pandas==0.24.2
# pip install pandas_ml

# Load pandas_ml which integrates pandas, scikit-learn and xgboost
import pandas_ml as pdml
# need to input numpy ndarray
cm = pdml.ConfusionMatrix(y_test.values, y_pred)
# Confusion matrix to pandas dataframe
cm_df = cm.to_dataframe(normalized=False, calc_sum=True,
sum_label='all')
# Confusion matrix partial results
print(cm_df.iloc[:12, :12])

# stats() generate overall and category-related indicators
perf_indx = cm.stats()
# save as collections (OrderedDict)
print(type(perf_indx))

# The key of an ordered dictionary structure, where cm is the same confusion matrix
print(perf_indx.keys())

# overall key is ordered dictionary structure
print(type(perf_indx['overall']))
# perf_indx['overall'].keys()

# overall indicators：
print(" acc:{}".format(perf_indx['overall']
['Accuracy']))

print(" acc95%:\n{}".format(perf_indx
['overall']['95% CI']))

print("Kappa:\n{}".format(perf_indx['overall']
['Kappa']))

# class key is pandas dataframe
print(type(perf_indx['class']))

# 26 letters (vertical) have each 26 categories (horizontal) Related indicators
print(perf_indx['class'].shape)

print(perf_indx['class'])

# Confusion matrix heat map visualization
import matplotlib.pyplot as plt
ax = cm.plot()
fig = ax.get_figure()
# fig.savefig('./_img/svc_rbf.png')