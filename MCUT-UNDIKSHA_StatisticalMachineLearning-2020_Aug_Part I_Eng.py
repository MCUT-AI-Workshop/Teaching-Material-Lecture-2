'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學)
Notes: This code is provided without warranty.
'''

### K-Means Clustering
import numpy as np

import pandas as pd

teens = pd.read_csv('./snsdata.csv', encoding = 'utf-8')

print(teens.dtypes)


print(teens.describe(include='all'))

teens.isnull().sum()

# print(sum(pd.isnull(teens['gender'])))


# print(teens['gender'].value_counts(dropna=False))


# print(teens['age'].describe())

# age = teens['age']

# list_age = []

# for i in age.index:

#    if  age[i] >= 13 and age[i] < 20:

#         list_age.append(age[i])

# df_age = pd.DataFrame(list_age)

 
# print(df_age.describe())

# # check our recoding work


# gender=teens['gender']

# list_gender = []

# list_no_gender=[]

# for i in gender.index:

#    if  gender[i]=='F' and gender[i]!=np.nan:

#         list_gender.append(1)

#         list_no_gender.append(0)

#    elif gender[i]=='M':

#             list_no_gender.append(0)

#    else:

#         list_gender.append(0)

#         list_no_gender.append(1)

# ## 

# teens=pd.concat([teens, pd.Series(list_gender).rename('female')], axis=1)

# teens=pd.concat([teens, pd.Series(list_no_gender).rename('no_gender')], axis=1)



# print(teens['female'].value_counts())

# print(teens['no_gender'].value_counts())



# np.mean(teens['age'])


# teens.groupby('gradyear').aggregate({'age': np.mean})


# ave_age=teens.groupby('gradyear')['age'].transform(np.mean)


# ave_age[:10]



# df_age_new=pd.DataFrame(teens['age'].fillna(teens.groupby('gradyear')['age'].transform(np.mean)))



# print(df_age_new['age'].describe())


interests = teens.loc[:, 'basketball':'drugs']


from sklearn import preprocessing

## 
teens_z = preprocessing.scale(interests)

teens_z = pd.DataFrame(teens_z)

teens_z.head(6)


from sklearn.cluster import KMeans

## 
mdl = KMeans(n_clusters = 5)

## 
## # Get the attributes and methods before fitting

pre = dir(mdl)

## 
## # Show a few of them

print(pre[51:56])

## 
## # Input standardized document-term matrix for model fitting

mdl.fit(teens_z)

## 
## # Get the attributes and methods after fitting

post = dir(mdl)

## 
## # # Show again

print(post[51:56])

## 
## # Difference set between 'post' and 'pre'

print(list(set(post) - set(pre)))


## # not run here

import pickle

filename = './_data/kmeans.sav'

# pickle.dump(mdl, open(filename, 'wb'))

res = pickle.load(open(filename, 'rb'))


pd.Series(mdl.labels_).value_counts()


mdl.labels_[:10]


## # Check the shape of cluster centers matrix

print(mdl.cluster_centers_.shape)

## 
## # Create a pandas DataFrame with keyworda for better presentation

cen = pd.DataFrame(mdl.cluster_centers_, index = range(5), columns=teens.iloc[:,4:40].columns)

print(cen)


## # Transpose the cluster centers matrix for plotting

ax = cen.T.plot()

## # x-axis ticks position setting

ax.set_xticks(list(range(36)))

## # x-axis labels setting (low-level plotting)

ax.set_xticklabels(list(cen.T.index), rotation=90)

fig = ax.get_figure()

fig.tight_layout()

## # fig.savefig('./_img/sns_lineplot.png')


teens = pd.concat([teens,pd.Series(mdl.labels_).rename('cluster')], axis=1)

## 


teens[['gender','age','friends','cluster']][0:5]


teens.groupby('cluster').aggregate({'age': np.mean})


teens.groupby('cluster').aggregate({'female': np.mean})


teens.groupby('cluster').aggregate({'friends': np.mean})


### Principal Component Analysis
import pandas as pd

import numpy as np

cell = pd.read_csv('segmentationOriginal.csv')


cell.head(2)

## 
cell.info() # RangeIndex, Columns, dtypes, memory type

## 
cell.shape


cell.columns.values # 119 variable names


cell.dtypes

## 
cell.describe(include = "all")


cell.isnull().any() # check NA by column


cell.isnull().values.any() # False, means no missing value ! Check the difference between above two !!!!

## 
#cell.isnull()

## #type(cell.isnull()) # pandas.core.frame.DataFrame, so .index, .column, and .values three important attributes

## 
#cell.isnull().values

#type(cell.isnull().values) # numpy.ndarray


cell.isnull().sum()


#cell['Case'].nunique()

cell['Case'].unique()

cell.Case.value_counts()

## #select the training set

cell_train = cell.loc[cell['Case']=='Train'] # same as cell[cell['Case']=='Train']

cell_train.head()

## 
cell['Case'][:10]

type(cell['Case']) # <class 'pandas.core.series.Series'>

cell[['Case']][:10]

type(cell[['Case']]) # <class 'pandas.core.frame.DataFrame'>


cell_data = cell_train.drop(['Cell','Class','Case'], axis=1)

cell_data.head()

## 
## # alternative way to do the same thing

cell_data = cell_train.drop(cell_train.columns[0:3], 1)

cell_data.head()


from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder # Encode labels with value between 0 and n_classes-1.

## 
## # label encoding

le_class = LabelEncoder().fit(cell['Class']) # 'PS': 0, 'WS': 1

Class_label = le_class.transform(cell['Class']) # 0: PS, 1: WS

Class_label.shape # (2019,)

## 
## # one-hot encoding

ohe_class = OneHotEncoder(sparse=False).fit(Class_label.reshape(-1,1)) # sparse : boolean, default=True Will return sparse matrix if set True else will return an array.

## #help(OneHotEncoder)

ohe_class.get_params()

## #{'categorical_features': 'all',

## # 'dtype': float,

## # 'handle_unknown': 'error',

## # 'n_values': 'auto',

## # 'sparse': False}

## #ohe_class.categorical_features

## 
Class_ohe = ohe_class.transform(Class_label.reshape(-1,1)) # (2019, 2)

## 
Class_label.reshape(-1,1).shape # (2019, 1) different to 1darray (2019,)

## 
Class_ohe.shape # (2019, 2) 2darray

Class_ohe


## # Fast way to do one-hot encoding or dummy encoding

Class_dum = pd.get_dummies(cell['Class'])

print (Class_dum.head())


print(cell_data.columns)

type(cell_data.columns) # pandas.core.indexes.base.Index


dir(pd.Series.str)

pd.Series(cell_data.columns).str.contains("Status").head() # logical indices after making cell_data.columns as pandas.Series

## #type(pd.Series(cell_data.columns).str.contains("Status")) # pandas.core.series.Series


cell_data.columns[pd.Series(cell_data.columns).str.contains("Status")] # again pandas.core.indexes.base.Index

## #type(cell_data.columns[pd.Series(cell_data.columns).str.contains("Status")]) # pandas.core.indexes.base.Index

## 
len(cell_data.columns[pd.Series(cell_data.columns).str.contains("Status")]) # 58 features with "Status"


cell_num = cell_data.drop(cell_data.columns[pd.Series(cell_data.columns).str.contains("Status")],axis=1)

cell_num.head()


from sklearn.decomposition import PCA

dr = PCA() # Principal Components Analysis

## 
## 
cell_pca = dr.fit_transform(cell_num) # PCA only for numeric

cell_pca

dir(dr)


dr.components_[:2] # [:2] can be removed, if you want to see more results of rotation matrix.

type(dr.components_) # numpy.ndarray

dr.components_.shape # (58, 58)


## # scree plot

dr.explained_variance_ratio_

import matplotlib.pyplot as plt

plt.plot(range(1, 26), dr.explained_variance_ratio_[:25], '-o')

plt.xlabel('# of components')

plt.ylabel('ratio of variance explained')


## # list(range(1,59))

## # range(1,59).tolist() # AttributeError: 'range' object has no attribute 'tolist'

## 
cell_dr = cell_pca[:,:5]

cell_dr

# pd.DataFrame(cell_dr).to_csv('cell_dr.csv')











