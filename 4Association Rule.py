# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:59:29 2020

@author: tom05
"""


### 線上音樂城關聯規則分析
## ------------------------------------------------------------------------
import pandas as pd
# 設定pandas 橫列與縱行結果呈現最大寬高值
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# 線上音樂城聆聽記錄載入
lastfm = pd.read_csv("./_data/lastfm.csv")
# 聆聽歷程長資料
print(lastfm.head())

# 檢視欄位資料型別，大多是類別變數
print(lastfm.dtypes)

# 統計各用戶線上聆聽次數
print(lastfm.user.value_counts()[:5])

# 獨一無二的用戶編號長度，共有15000 為用戶
print(lastfm.user.unique().shape)

# 各藝人被點閱次數
print(lastfm.artist.value_counts()[:5])

# 確認演唱藝人人數，共有1004 位藝人
print(lastfm.artist.unique().shape)

# 依用戶編號分組
grouped = lastfm.groupby('user')

# 檢視前兩組的子表，前兩位用戶各聆聽16 與29 位藝人專輯
print(list(grouped)[:2])

# 用戶編號有跳號現象
print(list(grouped.groups.keys())[:10])

# 以agg() 方法傳入字典，統計各使用者聆聽藝人數
numArt = grouped.agg({'artist': "count"})
print(numArt[5:10])

# 取出分組表藝人名稱一欄
grouped = grouped['artist']
# Python 串列推導，拆解分組資料為串列
music = [list(artist) for (user, artist) in grouped]

# 限於頁面寬度，取出交易記錄長度<3 的數據呈現巢狀串列的整理結果
print([x for x in music if len(x) < 3][:2])

from mlxtend.preprocessing import TransactionEncoder
# 交易資料格式編碼(同樣是宣告空模-> 擬合實模-> 轉換運用)
te = TransactionEncoder()
# 傳回numpy 二元值矩陣txn_binary
txn_binary = te.fit(music).transform(music)
# 檢視交易記錄筆數與品項數
print(txn_binary.shape)

# 讀者自行執行dir()，可以發現te 實模物件下有columns_ 屬性
# dir(te)
# 檢視部分品項名稱
print(te.columns_[15:20])

# numpy 矩陣組織為二元值資料框
df = pd.DataFrame(txn_binary, columns=te.columns_)
print(df.iloc[:5, 15:20])

# apriori 頻繁品項集探勘
from mlxtend.frequent_patterns import apriori
# pip install --trusted-host pypi.org mlxtend

# 挖掘時間長，因此記錄執行時間
# 可思考為何R 語言套件{arules} 的apriori() 快速許多？
import time
start = time.time()
freq_itemsets = apriori(df, min_support=0.01,
use_colnames=True)
end = time.time()
print(end - start)

# apply() 結合匿名函數統計品項集長度，並新增'length' 欄位於後
freq_itemsets['length'] = freq_itemsets['itemsets'].apply(lambda x: len(x))
# 頻繁品項集資料框，支持度、品項集與長度
print(freq_itemsets.head())

print(freq_itemsets.dtypes)

# 布林值索引篩選頻繁品項集
print(freq_itemsets[(freq_itemsets['length'] == 2)
& (freq_itemsets['support'] >= 0.05)])

# association_rules 關聯規則集生成
from mlxtend.frequent_patterns import association_rules
# 從頻繁品項集中產生49 條規則(生成規則confidence >= 0.5)
musicrules = association_rules(freq_itemsets,
metric="confidence", min_threshold=0.5)
print(musicrules.head())

# apply() 結合匿名函數統計各規則前提部長度
# 並新增'antecedent_len' 欄位於後
musicrules['antecedent_len'] = musicrules['antecedents'].apply(lambda x: len(x))
print(musicrules.head())

# 布林值索引篩選關聯規則
print(musicrules[(musicrules['antecedent_len'] > 0) &
(musicrules['confidence'] > 0.55)&(musicrules['lift'] > 5)])