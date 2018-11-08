import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

if __name__ == '__main__':
    train_data, test_data = load_data()
    add_miss_data(train_data)
