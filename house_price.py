import numpy as np
import pandas as pd

#读取数据
train_df = pd.read_csv('d:/train.csv', index_col=0)
test_df = pd.read_csv('d:/test.csv', index_col=0)

#检视原数据
print(train_df.head())

#拿出训练集SalePrice项，合并数据
prices = pd.DataFrame({'price':train_df['SalePrice'], 'log(price+1)':np.log1p(train_df['SalePrice'])})
y_train = np.log1p(train_df.pop('SalePrice'))
all_df = pd.concat((train_df, test_df), axis=0)
print(all_df.shape)
print(y_train.head())

#变量转化
print(all_df['MSSubClass'].dtypes)
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
all_df['MSSubClass'].value_counts()

#采用One-Hot方法将category类型numerical化
print(pd.get_dummies(all_df['MSSubClass'], prefix='MSSubClass').head())
all_dummy_df = pd.get_dummies(all_df)
print(all_dummy_df.head())

#缺失值填充
print(all_dummy_df.isnull().sum().sort_values(ascending=False).head())
mean_col = all_dummy_df.mean()
all_dummy_df = all_dummy_df.fillna(mean_col)
print(all_dummy_df.isnull().sum().sum())

#标准化numerical数据
numeric_cols = all_df.columns[all_df.dtypes != 'object']
print(numeric_cols)
numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means)/numeric_col_std

#建立模型
dumy_train_df = all_dummy_df.loc[train_df.index]
dumy_test_df = all_dummy_df.loc[test_df.index] #把数据分会训练集和测试集

#Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
X_train = dumy_train_df.values
X_test = dumy_test_df.values #转化为Numpy array,更加配sklearn

#调参数
alphas = np.logspace(-3, 2, 50)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(alphas, test_scores)
plt.title('Alpha vs CV Error') #大概alpha=10~20的时候，可以把score达到0.135左右

#Random Forest
from sklearn.ensemble import RandomForestRegressor
max_features = [.1, .3, .5, .7, .9, .99]
test_scores = []
for max_feature in max_features:
    clf = RandomForestRegressor(n_estimators=200, max_features=max_feature)
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
plt.plot(max_features, test_scores)
plt.title("Max Features vs CV Error") #RF最优值0.137

#Ensemble
ridge = Ridge(alpha=15)
rf = RandomForestRegressor(n_estimators=500, max_features=.3)
ridge.fit(X_train, y_train)
rf.fit(X_train, y_train)
#predit的值exp回去
y_ridge = np.expm1(ridge.predict(X_test))
y_rf = np.expm1(rf.predict(X_test))
y_final = (y_ridge+y_rf)/2

submission_df = pd.DataFrame(data={'ID':test_df.index, 'SalePrice':y_final})
print(submission_df.head(20)) #预测结果