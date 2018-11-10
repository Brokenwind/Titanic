import re
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

def get_combined_data():
    train_df_org = pd.read_csv('data/train.csv')
    test_df_org = pd.read_csv('data/test.csv')
    test_df_org['Survived']=0
    combined_train_test = train_df_org.append(test_df_org)
    return combined_train_test

def process_embarked(combined_train_test):
    '''
    '''
    combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
    combined_train_test['Embarked'] = pd.factorize(combined_train_test['Embarked'])[0]
    # df['column'] is a Series, but df[['column']] is a DataFrame
    emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'], prefix=combined_train_test[['Embarked']].columns[0])
    combined_train_test = pd.concat([combined_train_test, emb_dummies_df],axis=1)

    return combined_train_test

def process_sex(combined_train_test):
    '''
    '''
    combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]
    # df['column'] is a Series, but df[['column']] is a DataFrame
    sex_dummies_df = pd.get_dummies(combined_train_test['Sex'], prefix=combined_train_test[['Sex']].columns[0])
    combined_train_test = pd.concat([combined_train_test, sex_dummies_df],axis=1)

    return combined_train_test

def process_name(combined_train_test):
    combined_train_test['Title'] = combined_train_test['Name'].map(lambda x: re.compile(", (.*)\.").findall(x)[0])
    # map similar titile to one specified
    title_dict = {}
    title_dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    title_dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty')) 
    title_dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs')) 
    title_dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss')) 
    title_dict.update(dict.fromkeys(['Mr'], 'Mr')) 
    title_dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
    combined_train_test['Title'] = combined_train_test['Title'].map(title_dict)
    combined_train_test['Title'] = pd.factorize(combined_train_test['Title'])[0]
    title_dummies_df = pd.get_dummies(combined_train_test['Title'], prefix=combined_train_test[['Title']].columns[0])    
    combined_train_test = pd.concat([combined_train_test,title_dummies_df], axis=1)
    combined_train_test['Name_len'] = combined_train_test['Name'].map(len)

    return combined_train_test

def process_fare(combined_train_test):
    combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform(np.mean))

    return combined_train_test

if __name__ == '__main__':
    #train_data, test_data = load_data()
    #add_miss_data(train_data)
    combined_train_test = get_combined_data()
    #combined_train_test = process_embarked(combined_train_test)
    #combined_train_test = process_sex(combined_train_test)
    #combined_train_test = process_name(combined_train_test)
    combined_train_test = process_fare(combined_train_test)
    print(combined_train_test['Fare'])
