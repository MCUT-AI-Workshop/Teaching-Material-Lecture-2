# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:05:15 2020

@author: tom05
"""


### 銀行貸款風險管理案例
## ------------------------------------------------------------------------
import numpy as np
import pandas as pd
# 讀入UCI 授信客戶資料集
credit = pd.read_csv("./_data/germancredit.csv")
print(credit.shape)

# 檢視變數型別
print(credit.dtypes)

# 目標變數Default(已為0-1 值) 次數分佈
print(credit.Default.value_counts())

# 變數轉換字典target
target = {0: "Not Default", 1: "Default"}
credit.Default = credit.Default.map(target)

# 成批產製類別變數(dtype 為object) 的次數分佈表(存為字典結構)
# 先以邏輯值索引取出object 欄位名稱
col_cat = credit.columns[credit.dtypes == "object"]
# 逐步收納各類別變數次數統計結果用
counts_dict = {}
# 取出各欄類別值統計頻次
for col in col_cat:
    counts_dict[col] = credit[col].value_counts()
# 印出各類別變數次數分佈表
print(counts_dict)

# 代號與易瞭解名稱對照字典
print(dict(zip(credit.checkingstatus1.unique(),["< 0 DM",
"0-200 DM","> 200 DM","no account"])))

# 逐欄轉換易瞭解的類別名稱
credit.checkingstatus1 = credit.checkingstatus1.map(dict(zip
(credit.checkingstatus1.unique(),["< 0 DM","0-200 DM",
"> 200 DM","no account"])))
credit.history = credit.history.map(dict(zip(credit.history.
unique(),["good","good","poor","poor","terrible"])))
credit.purpose = credit.purpose.map(dict(zip(credit.purpose.
unique(),["newcar","usedcar","goods/repair","goods/repair",
"goods/repair","goods/repair","edu","edu","biz","biz"])))
credit.savings = credit.savings.map(dict(zip(credit.savings.
unique(),["< 100 DM","100-500 DM","500-1000 DM","> 1000 DM",
"unknown/no account"])))
credit.employ = credit.employ.map(dict(zip(credit.employ.
unique(),["unemployed","< 1 year","1-4 years","4-7 years",
"> 7 years"])))

# 基於篇幅考量，上面只顯示前五類別變數的轉換代碼，以下請讀者自行執行。
credit.status = credit.status.map(dict(zip(credit.status.unique(),["M/Div/Sep","F/Div/Sep/Mar","M/Single","M/Mar/Wid"])))
credit.others = credit.others.map(dict(zip(credit.others.unique(),["none","co-applicant","guarantor"])))
credit.property = credit.property.map(dict(zip(credit.property.unique(),["none","co_applicant","guarantor"])))
credit.otherplans = credit.otherplans.map(dict(zip(credit.otherplans.unique(),["bank","stores","none"])))
credit['rent'] = credit['housing'] == "A151"
credit['rent'].value_counts()
# del credit['housing']

credit.job = credit.job.map(dict(zip(credit.job.unique(),["unemployed","unskilled","skilled", "mgt/self-employed"])))
credit.tele = credit.tele.map(dict(zip(credit.tele.unique(),["none","yes"])))
credit.foreign = credit.foreign.map(dict(zip(credit.foreign.unique(),["foreign","german"])))
    
# 資料表內容較容易瞭解
print(credit.head())

# 授信客戶資料摘要統計表
print(credit.describe(include='all'))

# crosstab() 函數建支票存款帳戶狀況，與是否違約的二維列聯表
ck_f = pd.crosstab(credit['checkingstatus1'],
credit['Default'], margins=True)
# 計算相對次數
ck_f.Default = ck_f.Default/ck_f.All
ck_f['Not Default'] = ck_f['Not Default']/ck_f.All
print(ck_f)

# 儲蓄存款帳戶餘額狀況，與是否違約的二維列聯表
sv_f = pd.crosstab(credit['savings'],
credit['Default'], margins=True)
sv_f.Default = sv_f.Default/sv_f.All

sv_f['Not Default'] = sv_f['Not Default']/sv_f.All
print(sv_f)

# 與R 語言summary() 輸出相比，多了樣本數count 與標準差std
print(credit['duration'].describe())

print(credit['amount'].describe())

# 字串轉回0-1 整數值
inv_target = {"Not Default": 0, "Default": 1}
credit.Default = credit.Default.map(inv_target)

# 成批完成類別預測變數標籤編碼
from sklearn.preprocessing import LabelEncoder
# 先以邏輯值索引取出類別欄位名稱
col_cat = credit.columns[credit.dtypes == "object"]
# 宣告空模
le = LabelEncoder()
# 逐欄取出類別變數值後進行標籤編碼
for col in col_cat:
    credit[col] = le.fit_transform(credit[col].astype(str))

# 切分類別標籤向量y 與屬性矩陣X
y = credit['Default']
X = credit.drop(['Default'], axis=1)
# 切分訓練集及測試集，random_state 引數設定亂數種子
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.1, random_state=33)

# 訓練集類別標籤次數分佈表
Default_train = pd.DataFrame(y_train.value_counts(sort =
True))
# 計算與建立累積和欄位'cum_sum'
Default_train['cum_sum'] = Default_train['Default'].cumsum()
# 計算與建立相對次數欄位'perc'
tot = len(y_train)
Default_train['perc']=100*Default_train['Default']/tot
# 計算與建立累積相對次數欄位'cum_perc'
Default_train['cum_perc']=100*Default_train['cum_sum']/tot

### 限於篇幅，書本省略下面程式碼
# 測試集類別標籤次數分佈表
Default_test = pd.DataFrame(y_test.value_counts(sort=True))
# 計算與建立累積和欄位'cum_sum'
Default_test['cum_sum'] = Default_test['Default'].cumsum()
# 計算與建立相對次數欄位'perc'
Default_test['perc'] = 100*Default_test['Default']/len(y_test)
# 計算與建立累積相對次數欄位'cum_perc'
Default_test['cum_perc'] = 100*Default_test['cum_sum']/len(y_test)
###

# 比較訓練集與測試集類別標籤分佈
print(Default_train)
print(Default_test)

# 載入sklearn 套件的樹狀模型模組tree
from sklearn import tree
# 宣告DecisionTreeClassifier() 類別空模clf(未更改預設設定)
clf = tree.DecisionTreeClassifier()
# 傳入訓練資料擬合實模clf
clf = clf.fit(X_train,y_train)
# 預測訓練集標籤train_pred
train_pred = clf.predict(X_train)
print(' 訓練集錯誤率為{0}.'.format(np.mean(y_train !=
train_pred)))

# 預測測試集標籤test_pred
test_pred = clf.predict(X_test)
# 訓練集錯誤率遠低於測試集，過度配適的癥兆
print(' 測試集錯誤率為{0}.'.format(np.mean(y_test !=
test_pred)))

# print(clf.get_params())
keys = ['max_depth', 'max_leaf_nodes', 'min_samples_leaf']
print([clf.get_params().get(key) for key in keys])

# 再次宣告空模clf(更改上述三參數設定)、配適與預測
clf = tree.DecisionTreeClassifier(max_leaf_nodes = 10,
min_samples_leaf = 7, max_depth= 30)
clf = clf.fit(X_train,y_train)
train_pred = clf.predict(X_train)
print(' 訓練集錯誤率為{0}.'.format(np.mean(y_train !=
train_pred)))

# 過度配適情況已經改善
test_pred = clf.predict(X_test)
print(' 測試集錯誤率為{0}.'.format(np.mean(y_test !=
test_pred)))

n_nodes = clf.tree_.node_count
print(' 分類樹有{0} 個節點.'.format(n_nodes))

children_left = clf.tree_.children_left
s1 = ' 各節點的左子節點分別是{0}'
s2 = '\n{1}(-1 表葉子節點沒有子節點)。'
print(''.join([s1, s2]).format(children_left[:9],
children_left[9:]))

children_right = clf.tree_.children_right
s1 = ' 各節點的右子節點分別是{0}'
s2 = '\n{1}(-1 表葉子節點沒有子節點)。'
print(''.join([s1, s2]).format(children_right[:9],
children_right[9:]))

feature = clf.tree_.feature
s1 = ' 各節點分支屬性索引為(-2 表無分支屬性)'
s2 = '\n{0}。'
print(''.join([s1, s2]).format(feature))

threshold = clf.tree_.threshold
s1 = ' 各節點分支屬性門檻值為(-2 表無分支屬性門檻值)'
s2 = '\n{0}\n{1}\n{2}\n{3}。'
print(''.join([s1, s2]).format(threshold[:6],
threshold[6:12], threshold[12:18], threshold[18:]))

# 各節點樹深串列node_depth
node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
# 各節點是否為葉節點的真假值串列
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# 值組(節點編號, 父節點深度) 形成的堆疊串列, 初始化時只有根節點
stack = [(0, -1)]
# 從堆疊逐一取出資訊產生報表，堆疊最終會變空
while len(stack) > 0:
    node_i, parent_depth = stack.pop()
    # 自己的深度為父節點深度加1
    node_depth[node_i] = parent_depth + 1
    # 如果是測試節點(i.e. 左子節點不等於右子節點)，而非葉節點
    if (children_left[node_i] != children_right[node_i]):
    # 加左分枝節點，分枝節點的父節點深度正是自己的深度
        stack.append((children_left[node_i],parent_depth+1))
    # 加右分枝節點，分枝節點的父節點深度正是自己的深度
        stack.append((children_right[node_i],parent_depth+1))
    else:
    # is_leaves 原預設全為False，最後有True 有False
        is_leaves[node_i] = True

print(" 各節點的深度分別為：{0}".format(node_depth))

print(" 各節點是否為終端節點的真假值分別為：\n{0}\n{1}"
.format(is_leaves[:10], is_leaves[10:]))

print("%s 個節點的二元樹結構如下：" % n_nodes)
# 迴圈控制敘述逐一印出分類樹模型報表

for i in range(n_nodes):
    if is_leaves[i]:
        print("%snd=%s leaf nd."%(node_depth[i]*" ", i))
    else:
        s1 = "%snd=%s test nd: go to nd %s"
        s2 = " if X[:, %s] <= %s else to nd %s."
        print(''.join([s1, s2])
        % (node_depth[i] * " ",
        i,
        children_left[i],
        feature[i],
        threshold[i],
        children_right[i],
        ))

print()

# 載入Python 語言字串讀寫套件
from io import StringIO
# import pydot
import pydotplus
# 將樹tree 輸出為StringIO 套件的dot_data
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=
['checkingstatus1', 'duration', 'history', 'purpose',
'amount', 'savings', 'employ', 'installment', 'status',
'others', 'residence', 'property', 'age', 'otherplans',
'housing', 'cards', 'job', 'liable', 'tele', 'foreign', 'rent'],
class_names = ['Not Default', 'Default'])

# An alternative way on Windows
# import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/' # 裝到C:/Program Files (x86)/Graphviz2.38/

# dot_data 轉為graph 物件
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph 寫出pdf
# graph.write_pdf("credit.pdf")
print(graph)
# graph 寫出png
# graph.write_png('credit.png')
# 載入IPython 的圖片呈現工具類別Image(還有Audio 與Video)
# from IPython.core.display import Image
# Image(filename='credit.png')

### 酒品評點迴歸樹預測
## ------------------------------------------------------------------------
# Python 基本套件與資料集載入
import numpy as np
import pandas as pd
wine = pd.read_csv("./_data/whitewines.csv")

# 檢視變數型別
print(wine.dtypes)

# 葡萄酒資料摘要統計表
print(wine.describe(include='all'))

# 葡萄酒評點分數分佈
ax = wine.quality.hist()
ax.set_xlabel('quality')
ax.set_ylabel('frequency')
fig = ax.get_figure()
# fig.savefig("./_img/quality_hist.png")

# 切分屬性矩陣X 與類別標籤向量y
X = wine.drop(['quality'], axis=1)
y = wine['quality']
# 切分訓練集與測試集
X_train = X[:3750]
X_test = X[3750:]
y_train = y[:3750]
y_test = y[3750:]

from sklearn import tree
# 模型定義(未更改預設設定) 與配適
clf = tree.DecisionTreeRegressor()
# 儲存模型clf 參數值字典(因為直接印出會超出邊界)
dicp = clf.get_params()
# 取出字典的鍵，並轉為串列
dic = list(dicp.keys())
# 以字典推導分六次印出模型clf 的參數值
print({key:dicp.get(key) for key in dic[0:int(len(dic)/6)]})

# 第二次列印模型clf 參數值
print({key:dicp.get(key) for key in
dic[int(len(dic)/6):int(2*len(dic)/6)]})

# 第三次列印模型clf 參數值
print({key:dicp.get(key) for key in
dic[int(2*len(dic)/6):int(3*len(dic)/6)]})

# 第四次列印模型clf 參數值
print({key:dicp.get(key) for key in
dic[int(3*len(dic)/6):int(4*len(dic)/6)]})

# 第五次列印模型clf 參數值
print({key:dicp.get(key) for key in
dic[int(4*len(dic)/6):int(5*len(dic)/6)]})

# 第六次列印模型clf 參數值
print({key:dicp.get(key) for key in
dic[int(5*len(dic)/6):int(6*len(dic)/6)]})

# 迴歸樹模型配適
clf = clf.fit(X_train,y_train)
# 節點數過多(2123 個)，顯示節點過度配適
n_nodes = clf.tree_.node_count
print(' 迴歸樹有{0} 節點。'.format(n_nodes))

# 再次宣告空模clf(同上小節更改為R 語言套件{rpart} 的預設值)
clf = tree.DecisionTreeRegressor(max_leaf_nodes = 10,
min_samples_leaf = 7, max_depth= 30)
clf = clf.fit(X_train,y_train)
# 節點數19 個，顯示配適結果改善
n_nodes = clf.tree_.node_count
print(' 迴歸樹有{0} 節點。'.format(n_nodes))

# 預測訓練集酒質分數y_train_pred
y_train_pred = clf.predict(X_train)
# 檢視訓練集酒質分數的實際值分佈與預測值分佈
print(y_train.describe())

# 訓練集酒質預測分佈內縮
print(pd.Series(y_train_pred).describe())

# 預測測試集酒質分數y_test_pred
y_test_pred = clf.predict(X_test)
print(y_test.describe())

# 測試集酒質預測分佈內縮
print(pd.Series(y_test_pred).describe())

# 計算模型績效
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
print(' 訓練集MSE: %.3f, 測試集: %.3f' % (
mean_squared_error(y_train, y_train_pred),
mean_squared_error(y_test, y_test_pred)))

print(' 訓練集R^2: %.3f, 測試集R^2: %.3f' % (
r2_score(y_train, y_train_pred),
r2_score(y_test, y_test_pred)))