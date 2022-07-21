# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# Import keras libraries and packages
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

# Import Dataset

dataset = pd.read_csv('../../CSVs/Input n Capacity.csv')
X = dataset.iloc[:, 0:5].values
y = dataset.iloc[:, 5].values

skf = KFold(n_splits=5, shuffle=True)
fold_no = 0
traintot = 0
testtot = 0
for train_index,test_index in skf.split(dataset, y):
    train = dataset.iloc[train_index,:]
    test = dataset.iloc[test_index,:]

    Xtrain = train.drop(["Capacity(Ah)","SampleId"],axis=1)
    Ytrain = train["Capacity(Ah)"]

    dataB0005 = dataset.loc[dataset['SampleId'] == 'B0005']
    Xtest = dataB0005.drop(["Capacity(Ah)","SampleId"],axis=1)
    Ytest = dataB0005["Capacity(Ah)"]

    sc = StandardScaler()
    Xtrain_norm = sc.fit_transform(Xtrain)
    Xtest_norm = sc.transform(Xtest)

    # Initializing the ANN
    regressor = Sequential()

    # Adding the input layer and first hidden layer
    regressor.add(Dense(10 ,kernel_initializer = 'uniform',activation = 'relu',input_dim = 5))

    regressor.add(Dense(5 ,kernel_initializer = 'uniform',activation = 'tanh'))

    # Adding the output layer
    regressor.add(Dense(1,kernel_initializer = 'uniform',activation = 'linear'))

    # Compiling the ANN
    regressor.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics= ['mean_absolute_error'])

    # Fitting the ANN in the Training set
    regressor.fit(Xtrain_norm, Ytrain, batch_size = 1,epochs = 30)
    # Predicting the Test set result
    ytrain_pred = regressor.predict(Xtrain_norm)
    trainscore = r2_score(Ytrain, ytrain_pred)
    ytest_pred = regressor.predict(Xtest_norm)
    testscore = r2_score(Ytest, ytest_pred)
    testrmsescore = np.sqrt(mean_squared_error(Ytest,ytest_pred))

    print(testrmsescore)
    traintot += trainscore
    testtot += testscore
    fold_no += 1

    # listt = X_005[0]
    # init_capacity = 0.7 * regressor.predict(listt)
    # x_inp = []
    # for i in range(1,len(ytest_pred)+1):
    #     x_inp.append(i)
    # plt.plot(y_train, color='magenta')
    print(pd.DataFrame({'Actual':Ytest,'Predicted':np.ravel(ytest_pred)}).iloc[:10].to_string(index=False))
    plt.plot(Xtest.iloc[:,0], Ytest, color='red')
    plt.plot(Xtest.iloc[:,0], ytest_pred, color='blue')
    threshold = 1.5
    plt.axhline(threshold, color='yellow', linestyle='--')
    plt.title('RUL Prediction')
    plt.xlabel('Cycle')
    plt.ylabel('Capacity (Ah)')
    plt.show()

    print('Training accuracy:', trainscore)
    print('Testing accuracy:', testscore)

print('Average training accuracy:', traintot/fold_no)

print('Average testing accuracy:', testtot/fold_no)

##############
