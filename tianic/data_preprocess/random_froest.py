from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tianic.data_preprocess.DataReadAndClean import *


def feature_selection():
    data_train = readSourceData()
    data_train = dataPreprocess(data_train)
    predictors = ["Pclass", "Sex", "Age", "NewFare", "Embarked",'IsAlone']

    # Perform feature selection
    selector = SelectKBest(f_classif, k=5)
    selector.fit(data_train[predictors], data_train["Survived"])

    # Plot the raw p-values for each feature,and transform from p-values into scores
    scores = -np.log10(selector.pvalues_)

    # Plot the scores.   See how "Pclass","Sex","Title",and "Fare" are the best?
    plt.bar(range(len(predictors)),scores)
    plt.xticks(range(len(predictors)),predictors, rotation='vertical')
    plt.show()

def randomForest(data_train):
    # Pick only the four best features.
    predictors = ["Pclass", "Sex", "NewFare", "Embarked", 'IsAlone']
    X_train, X_test, Y_train, Y_test = train_test_split(data_train[predictors], data_train['Survived'], test_size=0.2)
    alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
    alg.fit(X_train, Y_train)
    Y_predict = alg.predict(X_test)
    acc = sum(Y_predict == Y_test) / len(Y_predict)
    return acc

data_train = readSourceData();
data_train = dataPreprocess(data_train)
total = 0
for i in range(5):
    total = total + randomForest(data_train)
print((total/5))