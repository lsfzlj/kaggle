import pandas as pd

#数据读取
from sklearn.cluster import KMeans


def readSourceData():
    data_train = pd.read_csv("../data/train.csv")
    return data_train

def dataPreprocess(df):
    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1

    # 由于 Embarked中有两个数据未填充，需要先将数据填满
    df['Embarked'] = df['Embarked'].fillna('S')
    # 部分年龄数据未空， 填充为 均值
    df['Age'] = df['Age'].fillna(df['Age'].median())

    df.loc[df['Embarked']=='S', 'Embarked'] = 0
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2

    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['IsAlone'] = 0
    df.loc[df['FamilySize']==0,'IsAlone'] = 1
    df.drop('FamilySize',axis = 1)
    df.drop('Parch',axis=1)
    df.drop('SibSp',axis=1)
    return  fare_kmeans(df)

def fare_kmeans(data_train):
    clusters = KMeans(n_clusters=5)
    clusters.fit(data_train['Fare'].values.reshape(-1, 1))
    predict = clusters.predict(data_train['Fare'].values.reshape(-1, 1))
    data_train['NewFare'] = predict
    data_train.drop('Fare',axis=1)
    # print(data_train[['NewFare','Survived']].groupby(['NewFare'],as_index=False).mean())
    # print(" "  + str(clusters.inertia_))
    return data_train






