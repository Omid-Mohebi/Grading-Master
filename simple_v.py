import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
X_train = train_data.iloc[:, 4:-1].to_numpy()
Y_train = train_data.iloc[:,-1].to_numpy()
X_test = test_data.iloc[:, 4:].to_numpy()

def one_hard_encoding(data):
    classes = set(data)
    class_dict = {}
    N = len(data)
    K = len(classes)
    ohe = np.zeros((N, K))
    for i, clas in zip(range(K), classes):
        class_dict[clas] = i
    for i in range(N):
        ohe[i, class_dict[data[i]]] = 1
    return ohe


X_train = np.concatenate([one_hard_encoding(X_train[:, 0]), X_train[:, 1:4], one_hard_encoding(X_train[:, 4]), X_train[:, 5:]], axis=1).astype(np.float32)
X_test = np.concatenate([one_hard_encoding(X_test[:, 0]), X_test[:, 1:4], one_hard_encoding(X_test[:, 4]), X_test[:, 5:]], axis=1).astype(np.float32)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, Y_train)
model.score(X_train, Y_train)

