'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), Ming Chi University of Technology (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), National Taipei University of Business (國立臺北商業大學); the Chinese Academy of R Software (中華R軟體學會); and the Data Science & Business Applications Association of Taiwan (臺灣資料科學與商業應用協會)
Notes: This code is provided without warranty.
'''

# First of all, just like what you do with any other dataset, you are going to import the Boston Housing dataset and store it in a variable called boston. To import it from scikit-learn you will need to run this snippet.
from sklearn.datasets import load_boston
boston = load_boston()

#The boston variable itself is a dictionary, so you can check for its keys using the .keys() method.
print(type(boston))

print(boston.keys())

# You can easily check for its shape by using the boston.data.shape attribute, which will return the size of the dataset.
print(boston.data.shape)

# As you can see it returned (506, 13), that means there are 506 rows of data with 13 columns. Now, if you want to know what the 13 columns are, you can simply use the .feature_names attribute and it will return the feature names.
print(boston.feature_names)

# The description of the dataset is available in the dataset itself. You can take a look at it using .DESCR.
print(boston.DESCR)

# Now let’s convert it into a pandas DataFrame! For that you need to import the pandas library and call the DataFrame() function passing the argument boston.data. To label the names of the columns, please use the .columnns attribute of the pandas DataFrame and assign it to boston.feature_names.
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

type(boston.data)
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

# Explore the top 5 rows of the dataset by using head() method on your pandas DataFrame.
data.head()

# You'll notice that there is no column called PRICE in the DataFrame. This is because the target column is available in another attribute called boston.target. Append boston.target to your pandas DataFrame.
data['PRICE'] = boston.target

# Run the .info() method on your DataFrame to get useful information about the data.
data.info()

# Note that describe() only gives summary statistics of columns which are continuous in nature and not categorical.
data.describe()

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Separate the target variable and rest of the variables using .iloc to subset the data.
X, y = data.iloc[:, :-1],data.iloc[:, -1]

# Now you will convert the dataset into an optimized data structure called Dmatrix that XGBoost supports and gives it acclaimed performance and efficiency gains. You will use this later in the tutorial.
data_dmatrix = xgb.DMatrix(data=X,label=y)

# Now, you will create the train and test set for cross-validation of the results using the train_test_split function from sklearn's model_selection module with test_size size equal to 20% of the data. Also, to maintain reproducibility of the results, a random_state is also assigned.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# The next step is to instantiate an XGBoost regressor object by calling the XGBRegressor() class from the XGBoost library with the hyper-parameters passed as arguments. For classification problems, you would have used the XGBClassifier() class.
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

# Fit the regressor to the training set and make predictions on the test set using the familiar .fit() and .predict() methods.
xg_reg.fit(X_train,y_train)
# WARNING: /Users/travis/build/dmlc/xgboost/src/objective/regression_obj.cu:174: reg:linear is now deprecated in favor of reg:squarederror.

preds = xg_reg.predict(X_test)

# Compute the rmse by invoking the mean_sqaured_error function from sklearn's metrics module.
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# You will use these parameters to build a 3-fold cross validation model by invoking XGBoost's cv() method and store the results in a cv_results DataFrame. Note that here you are using the Dmatrix object you created before.
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1, 'max_depth': 5, 'alpha': 10}


cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)

# cv_results contains train and test RMSE metrics for each boosting round.
cv_results.head()

cv_results.tail()

# Extract and print the final boosting round metric.
print((cv_results["test-rmse-mean"]).tail(1))

# Set the number of boosting round to 10
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

# Another way to visualize your XGBoost models is to examine the importance of each feature column in the original dataset within the model.

# One simple way of doing this involves counting the number of times each feature is split on across all boosting rounds (trees) in the model, and then visualizing the result as a bar graph, with the features ordered according to how many times they appear. XGBoost has a plot_importance() function that allows you to do exactly this.
import matplotlib.pyplot as plt

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

# As you can see the feature RM has been given the highest importance score among all the features. Thus XGBoost also gives you a way to do Feature Selection. Isn't this brilliant?

# Conclusion
# You have reached the end of this tutorial. I hope this might have or will help you in some way or the other. You started off with understanding how Boosting works in general and then narrowed down to XGBoost specifically. You also practiced applying XGBoost on an open source dataset and along the way you learned about its hyper-parameters, doing cross-validation, visualizing the trees and in the end how it can also be used as a Feature Selection technique. Whoa!! that's something for starters, but there is so much to explore in XGBoost that it can't be covered in a single tutorial. If you would like to learn more, be sure to take a look at our Extreme Gradient Boosting with XGBoost course on DataCamp.