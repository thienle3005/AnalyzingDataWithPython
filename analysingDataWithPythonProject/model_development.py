from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

path='./file/clean_df.csv'
df = pd.read_csv(path)

### Linear Regression and Multiple Linear Regression
## Linear Regression
# lm = LinearRegression()
# X = df[['highway-mpg']]
# Y = df['price']
# lm.fit(X,Y)
# Yhat = lm.predict(X)
# print(lm.intercept_)
# print(lm.coef_)
# print(Yhat[0:5]) #predict out put
# 𝑌ℎ𝑎𝑡=𝑎+𝑏𝑋
# price = 38423.31 - 821.73 x highway-mpg

# lm1 = LinearRegression()
# lm1.fit(df[['highway-mpg']], df[['price']])
# print(lm1.coef_ )
## Multiple Linear Regression
lm = LinearRegression()
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])
# 𝑌ℎ𝑎𝑡=𝑎+𝑏1𝑋1+𝑏2𝑋2+𝑏3𝑋3+𝑏4𝑋4
# predictedvalue(Yhat)
# print(lm.intercept_)
# print(lm.coef_)
# Price = -15678.742628061467 + 52.65851272 x horsepower + 4.69878948 x curb-weight + 81.95906216 x engine-size + 33.58258185 x highway-mpg


###1 Model Evaluation using Visualization¶
## visualize Horsepower as potential predictor variable of price
# width = 12
# height = 10
# plt.figure(figsize=(width, height))
# sns.regplot(x="highway-mpg", y="price", data=df)
# plt.ylim(0,)
# plt.show()

#2 visualize peak-rpm as potential predictor variable of price
# width = 12
# height = 10
# plt.figure(figsize=(width, height))
# sns.regplot(x="peak-rpm", y="price", data=df)
# plt.ylim(0,)
# plt.show()
## Residual Plot
# width = 12
# height = 10
#
# # plt.figure(figsize=(width, height))
# # sns.residplot(df['highway-mpg'], df['price'])
# # plt.show()

##3 Multiple Linear Regression
# width = 12
# height = 10
# Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
# lm = LinearRegression()
# lm.fit(Z, df['price'])
# Y_hat = lm.predict(Z)
# plt.figure(figsize=(width, height))
# ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
# sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
# plt.title('Actual vs Fitted Values for Price')
# plt.xlabel('Price (in dollars)')
# plt.ylabel('Proportion of Cars')
#
# plt.show()
# plt.close()

##Polynomial Regression and Pipelines

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    # ax.set_axis_bgcolor((0.898, 0.898, 0.898)) derepcated
    ax.set_facecolor((0.898, 0.898, 0.898))

    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


# x = df['highway-mpg']
# y = df['price']
# # Here we use a polynomial of the 3rd order (cubic)
# # f = np.polyfit(x, y, 3)
# # Here we use a polynomial of the 3rd order (cubic)
# f = np.polyfit(x, y, 11)
# p = np.poly1d(f)
# PlotPolly(p,x,y, 'highway-mpg')

##perform a polynomial transform on multiple features
# Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
# y = df['price']
# pr = PolynomialFeatures(degree=2)
# Z_pr = pr.fit_transform(Z)
# print(Z.shape) #(201, 4)
# print(Z_pr.shape) #(201, 15)
# Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
# pipe=Pipeline(Input)
# pipe.fit(Z,y)
# ypipe=pipe.predict(Z)
# print(ypipe[0:10])

###4 Measures for In-Sample Evaluation
# Model 1: Simple Linear Regression
# price = 38423.31 - 821.73 x highway-mpg
#highway_mpg_fit
# lm = LinearRegression()
X = df[['highway-mpg']]
Y = df['price']
lm.fit(X, Y)
# # calculate the R^2
# print("Simple Linear Regression R^2:  " + str(lm.score(X, Y))) #~ 49.659%
# # ## calculate the MSE
# Yhat=lm.predict(X)
# # ## mean_squared_error(Y_true, Y_predict)
# print("Simple Linear Regression MSE:  " + str(mean_squared_error(df['price'], Yhat)))
#
# ###5 Model 2: Multiple Linear Regression
# # Price = -15678.742628061467 + 52.65851272 x horsepower + 4.69878948 x curb-weight + 81.95906216 x engine-size + 33.58258185 x highway-mpg
# # calculate the R^2
# # fit the model
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z, df['price'])
# Find the R^2
print( "Multiple Linear Regression R^2: " + str(lm.score(Z, df['price'])))  #that ~ 80.896 %
# calculate the MSE
Y_predict_multifit = lm.predict(Z)
print("Multiple Linear Regression MSE: " + str(mean_squared_error(df['price'], Y_predict_multifit)))
#
# ###6 Model 3: Polynomial Fit
# poly = PolynomialFeatures(degree = 3)
# X_poly = poly.fit_transform(X)
#
# poly.fit(X_poly, Y)
# lin2 = LinearRegression()
# lin2.fit(X_poly, Y)
# Ypred = lin2.predict(X_poly)
# r2 = r2_score(Y,Ypred) #0.651793603702672
# print("Polynomial Fit R^2:             " + str(r2))
# print("Polynomial Fit MSE:             " + str(mean_squared_error(df['price'], Ypred)))


##5 Multiple Linear Regression
# lm = LinearRegression()
# Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
# lm.fit(Z, df['price'])
# #𝑌ℎ𝑎𝑡=𝑎+𝑏1𝑋1+𝑏2𝑋2+𝑏3𝑋3+𝑏4𝑋4
# #predictedvalue(Yhat)
# print(lm.intercept_)
# print(lm.coef_)
# Price = -15678.742628061467 + 52.65851272 x horsepower + 4.69878948 x curb-weight + 81.95906216 x engine-size + 33.58258185 x highway-mpg

### prediction and decision Making

new_input = np.arange(1,100,1).reshape(-1,1)
lm.fit(X,Y)
yhat = lm.predict(new_input)
plt.plot(new_input,yhat)
plt.show()


