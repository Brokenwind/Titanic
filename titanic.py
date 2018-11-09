import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

def load_data():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    train_data.info()
    print("-" * 40)
    test_data.info()
    return train_data, test_data

def add_miss_data(train_data):
    embarked_mode = train_data['Embarked'].dropna().mode().values
    print(embarked_mode)
    # 如果该属性相对学习来说不是很重要，可以对缺失值赋均值或者众数
    train_data['Embarked'][train_data['Embarked'].isnull()] = embarked_mode
    # 可以赋一个代表缺失的值，比如‘U0’。因为缺失本身也可能代表着一些隐含信息
    train_data['Cabin'] = train_data['Cabin'].fillna('U0')
    # 使用回归 随机森林等模型来预测缺失属性的值。因为Age在该数据集里是一个相当重要的特征
    age_df = train_data[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
    age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
    age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
    X = age_df_notnull.values[:,1:]
    Y = age_df_notnull.values[:,0]
    # use RandomForestRegression to train data
    RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    RFR.fit(X,Y)
    predictAges = RFR.predict(age_df_isnull.values[:,1:])
    train_data.loc[train_data['Age'].isnull(), ['Age']]= predictAges
    
    return train_data

if __name__ == '__main__':
    train_data, test_data = load_data()
    add_miss_data(train_data)
