from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from tianic.data_preprocess import DataReadAndClean
from tianic.data_preprocess.DataReadAndClean import dataPreprocess


def data_process_onehot(df):
    #copy_df = df.copy()
    train_Embarked = df["Embarked"].values.reshape(-1,1)

    onehot_encoder = OneHotEncoder(sparse=False)
    train_OneHotEncoded = onehot_encoder.fit_transform(train_Embarked)
    df["EmbarkedS"] = train_OneHotEncoded[:, 0]
    df["EmbarkedC"] = train_OneHotEncoded[:, 1]
    df["EmbarkedQ"] = train_OneHotEncoded[:, 2]
    return df

def linearRegression(df):
    predictors = ['Pclass', 'Sex', 'Age', 'IsAlone', 'NewFare', 'Embarked']
    #predictors = ['Pclass', 'Sex', 'Age', 'IsAlone', 'NewFare', 'EmbarkedS','EmbarkedC','EmbarkedQ']

    alg = LinearRegression()
    X = df[predictors]
    Y = df['Survived']
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

    # 打印 训练集 测试集 样本数量
    print (X_train.shape)
    print (Y_train.shape)
    print (X_test.shape)
    print (Y_test.shape)

    # 进行拟合
    alg.fit(X_train, Y_train)

    print (alg.intercept_)
    print (alg.coef_)

    Y_predict = alg.predict(X_test)
    Y_predict[Y_predict >= 0.5 ] = 1
    Y_predict[Y_predict < 0.5] = 0
    acc = sum(Y_predict==Y_test) / len(Y_predict)
    return acc

data_train = DataReadAndClean.readSourceData()
data_train = dataPreprocess(data_train)
#data_train = data_process_onehot(data_train)
precent = linearRegression(data_train)
total = 0
for num in range(5):
    total = total + linearRegression(data_train)
print((total/5))

