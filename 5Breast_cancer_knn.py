import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
plt.style.use('ggplot')
pd.set_option('display.max_columns', 500)

# Load csv from your local drive into a Dataframe df

# df = pd.read_csv('uci-breast-cancer.csv')
# df = df[~df.Bare_Nuclei.isin(['?'])]
# df = df.astype(float)
df = pd.read_csv('wisconsin_breast_cancer.csv')
df = df.dropna()

df.info()
df.describe()

X = df.drop(labels='class',axis=1)
# Data model will be made on this dataset
y = df['class'] # Target data

#Splitting data into training set (70%) and testing set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Making circle of region with k=3, we can increase or decrease this
knn = neighbors.KNeighborsClassifier(n_neighbors = 3)

#Training the machine with our model
knn_model_1 = knn.fit(X_train, y_train)
print('k-NN accuracy for test set: %f' % knn_model_1.score(X_test, y_test)) # 0.634146

y_true, y_pred = y_test, knn_model_1.predict(X_test)
print(classification_report(y_true, y_pred))


pd.DataFrame.hist(df.iloc[:,1:10], figsize = [15,15])

from sklearn.preprocessing import scale

Xs = scale(X)

Xs_train, Xs_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3, random_state=42)

knn_model_2 = knn.fit(Xs_train, y_train)

print('k-NN score for test set: %f' % knn_model_2.score(Xs_test, y_test))
print('k-NN score for training set: %f' % knn_model_2.score(Xs_train, y_train))

y_true, y_pred = y_test, knn_model_2.predict(Xs_test)

