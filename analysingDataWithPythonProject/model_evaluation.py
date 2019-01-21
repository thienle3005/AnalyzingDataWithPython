import pandas as pd
import numpy as np
from IPython.display import display
# from IPython.html import widgets
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

path = './file/clean_df.csv'
df = pd.read_csv(path)
df = df._get_numeric_data()


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    # training data
    # testing data
    # lr:  linear regression object
    # poly_transform:  polynomial transformation object

    xmax = max([xtrain.values.max(), xtest.values.max()])

    xmin = min([xtrain.values.min(), xtest.values.min()])

    x = np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()


###1 Training and Testing 15%
y_data = df['price']
# drop price data in x data
x_data = df.drop('price', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
## x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

lre = LinearRegression()
lre.fit(x_train[['horsepower']], y_train)
# # Calculate the R^2 on the test data:
# test =lre.score(x_test[['horsepower']],y_test)
# print(test)
# train = lre.score(x_train[['horsepower']],y_train)
# print(train)
### Training and Testing 90%
# x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.9, random_state=0)
# lre.fit(x_train1[['horsepower']],y_train1)
# train = lre.score(x_train1[['horsepower']],y_train1)
# print(train)
# test =lre.score(x_test1[['horsepower']],y_test1)
# print(test)
## Cross-validation Score
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
# calculate the average and standard deviation of our estimate

# print("The mean of the folds are", Rcross.mean(),"and the standard deviation is" ,Rcross.std())
# negative squared error
# -1*cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')
# yhat=cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
### Overfitting, Underfitting, and model selection
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
#
# Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution '
# DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
#
# Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
# DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
### Overfitting
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
poly = LinearRegression()
poly.fit(x_train_pr, y_train)
yhat = poly.predict(x_test_pr)
# print(yhat[0:5])
# print("Predicted values:", yhat[0:4])
# print("True values:",y_test[0:4].values)
# PollyPlot(x_train[['horsepower']],x_test[['horsepower']],y_train,y_test,poly,pr)
# R^2 of the train
# train =poly.score(x_train_pr, y_train)
# print(train)
# test =poly.score(x_test_pr, y_test)
# print(test)

# Rsqu_test = []
# order = [1, 2, 3, 4]
# for n in order:
#     pr = PolynomialFeatures(degree=n)
#
#     x_train_pr = pr.fit_transform(x_train[['horsepower']])
#
#     x_test_pr = pr.fit_transform(x_test[['horsepower']])
#
#     lr.fit(x_train_pr, y_train)
#
#     Rsqu_test.append(lr.score(x_test_pr, y_test))
#
# plt.plot(order, Rsqu_test)
# plt.xlabel('order')
# plt.ylabel('R^2')
# plt.title('R^2 Using Test Data')
# plt.text(3, 0.75, 'Maximum R^2 ')
#
#
# def f(order, test_data):
#     x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
#     pr = PolynomialFeatures(degree=order)
#     x_train_pr = pr.fit_transform(x_train[['horsepower']])
#     x_test_pr = pr.fit_transform(x_test[['horsepower']])
#     poly = LinearRegression()
#     poly.fit(x_train_pr, y_train)
#     PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly, pr)
#
#
# interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))

### Ridge regression
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
RigeModel=Ridge(alpha=0.1)

RigeModel.fit(x_train_pr,y_train)
yhat=RigeModel.predict(x_test_pr)
# print('predicted:', yhat[0:4])
# print('test set :', y_test[0:4].values)

Rsqu_test=[]
Rsqu_train=[]
dummy1=[]
ALFA=5000*np.array(range(0,10000))
# for alfa in ALFA:
RigeModel=Ridge(alpha=alfa)
RigeModel.fit(x_train_pr,y_train)
Rsqu_test.append(RigeModel.score(x_test_pr,y_test))
Rsqu_train.append(RigeModel.score(x_train_pr,y_train))

width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(ALFA,Rsqu_test,label='validation data  ')
plt.plot(ALFA,Rsqu_train,'r',label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()


### Grid Search
# parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000,10000,100000,100000]}]
# RR=Ridge()
# Grid1 = GridSearchCV(RR, parameters1,cv=4)
# Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_data)
# BestRR=Grid1.best_estimator_
# BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']],y_test)