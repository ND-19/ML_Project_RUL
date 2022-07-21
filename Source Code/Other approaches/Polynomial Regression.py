# Polynomial Regression

# Importing the libraries
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

# Importing the dataset
dataset = pd.read_csv('../../CSVs/Input n Capacity.csv')
X = dataset.iloc[:, 0:5].values
y = dataset.iloc[:, 5].values

skf = KFold(n_splits=5, shuffle=True)
fold_no = 0
traintot = 0
testtot = 0
trainrmsetot = 0
testrmsetot = 0
for train_index,test_index in skf.split(dataset, y):
    train = dataset.iloc[train_index,:]
    test = dataset.iloc[test_index,:]

    Xtrain = train.drop(["Capacity(Ah)","SampleId"],axis=1)
    Ytrain = train["Capacity(Ah)"]

    Xtest = test.drop(["Capacity(Ah)","SampleId"],axis=1)
    Ytest = test["Capacity(Ah)"]

    lin_reg = LinearRegression()
    lin_reg.fit(Xtrain,Ytrain)
 
    ytrain_pred = lin_reg.predict(Xtrain)
    trainscore = r2_score(Ytrain, ytrain_pred)
    trainrmsescore = np.sqrt(mean_squared_error(Ytrain,ytrain_pred))
    ytest_pred = lin_reg.predict(Xtest)
    testscore = r2_score(Ytest, ytest_pred)
    testrmsescore = np.sqrt(mean_squared_error(Ytest,ytest_pred))

    traintot += trainscore
    testtot += testscore
    trainrmsetot += trainrmsescore
    testrmsetot += testrmsescore
    fold_no += 1

    if fold_no == 1 :
        for j in range(len(Xtest.columns)) :
            if(j!=3) :
                xlabel= Xtest.columns[j]
                mp, bp = np.polyfit(Xtest.iloc[:,j], ytest_pred, 1)
                plt.scatter(Xtest.iloc[:,j], ytest_pred, color = 'red')
                plt.plot(Xtest.iloc[:,j], mp*Xtest.iloc[:,j] + bp, color='blue')
                plt.title('Capacity vs {} (Linear Regression)'.format(xlabel))
                plt.xlabel('{}'.format(xlabel))
                plt.ylabel('Capacity')
                plt.show()

    print("Linear Regression\n")
    print('Fold no',fold_no,'\n')
    print('Number of training examples',len(train.index),'\n')
    print('Number of testing examples',len(test.index),'\n')
    print(pd.DataFrame({'Actual':Ytest,'Predicted':np.ravel(ytest_pred)}).iloc[:10].to_string(index=False))
    print('Training accuracy:', trainscore)
    print('Training R2score:', trainrmsescore)
    print('Testing accuracy:', testscore)
    print('Testing R2score:', testrmsescore)
   
print('Average training accuracy:', traintot/fold_no)
print('Average training root mean squared error:', trainrmsetot/(fold_no))
print('Average testing accuracy:', testtot/fold_no)
print('Average testing root mean squared error:', testrmsetot/(fold_no))

skf = KFold(n_splits=5, shuffle=True)
fold_no = 0
traintot = 0
testtot = 0
trainrmsetot = 0
testrmsetot = 0
for train_index,test_index in skf.split(dataset, y):
    train = dataset.iloc[train_index,:]
    test = dataset.iloc[test_index,:]

    Xtrain = train.drop(["Capacity(Ah)","SampleId"],axis=1)
    Ytrain = train["Capacity(Ah)"]

    Xtest = test.drop(["Capacity(Ah)","SampleId"],axis=1)
    Ytest = test["Capacity(Ah)"]

    poly_reg = PolynomialFeatures(degree = 5)
    X_poly = poly_reg.fit_transform(Xtrain)
    poly_reg.fit(X_poly, Ytrain)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, Ytrain)
 
    ytrain_pred = lin_reg_2.predict(poly_reg.fit_transform(Xtrain))
    trainscore = r2_score(Ytrain, ytrain_pred)
    trainrmsescore = np.sqrt(mean_squared_error(Ytrain,ytrain_pred))
    ytest_pred = lin_reg_2.predict(poly_reg.fit_transform(Xtest))
    testscore = r2_score(Ytest, ytest_pred)
    testrmsescore = np.sqrt(mean_squared_error(Ytest,ytest_pred))

    traintot += trainscore
    testtot += testscore
    trainrmsetot += trainrmsescore
    testrmsetot += testrmsescore
    fold_no += 1

    if fold_no == 1 :
        for j in range(len(Xtest.columns)) :
            if(j != 3) :
                xlabel= Xtest.columns[j]
                X_sort = np.sort(Xtest.iloc[:,j])
                fit = np.polyfit(Xtest.iloc[:,j], ytest_pred, 3)
                fit_fn = np.poly1d(fit)
                plt.scatter(Xtest.iloc[:,j], ytest_pred, color = 'red')
                plt.plot(Xtest.iloc[:,j], ytest_pred, 'k.', X_sort, fit_fn(X_sort), 'b', linewidth=1)
                # plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
                plt.title('Capacity vs {} (Polynomial Regression)'.format(xlabel))
                plt.xlabel('{}'.format(xlabel))
                plt.ylabel('Capacity')
                plt.show()

    print("Polynomial Regression\n")
    print('Fold no',fold_no,'\n')
    print('Number of training examples',len(train.index),'\n')
    print('Number of testing examples',len(test.index),'\n')
    print(pd.DataFrame({'Actual':Ytest,'Predicted':np.ravel(ytest_pred)}).iloc[:10].to_string(index=False))
    print('Training accuracy:', trainscore)
    print('Training Rmsescore:', trainrmsescore)
    print('Testing accuracy:', testscore)
    print('Testing Rmsescore:', testrmsescore)
   
print('Average training accuracy:', traintot/fold_no)
print('Average training root mean squared error:', trainrmsetot/(fold_no))
print('Average testing accuracy:', testtot/fold_no)
print('Average testing root mean squared error:', testrmsetot/(fold_no))
