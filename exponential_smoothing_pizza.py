"""
We followed the tutorial by C hubbs to create this model
@author:https://www.datahubbs.com/forecasting-with-seasonality/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
from sklearn.linear_model import LinearRegression


def holt_winter_model():
   
    pizza=pd.read_csv('pizza_total.csv',sep=',')
    #drop removes data
    pizza = pizza.reset_index()
    pizza.head()
    
    # Split data
    split = int(0.70 * len(pizza))
    train = pizza[:split]
    test = pizza[split:]
    
    alpha=0.2
    beta=0.08
    gamma=0.1
    
    lr=LinearRegression(fit_intercept=True)
    X=train.index.values.reshape(-1,1)
    y=train.y.values.reshape(-1,1)
    lr.fit(X,y)
    
    a_hat=np.array([lr.intercept_[0]])
    b_hat=np.array([lr.coef_[0][0]])
    
    p=12
    F_fit=np.ones(p)
    
    y_hat = np.array([(a_hat + b_hat) * F_fit[0]])
    
    for i in range(1, len(y)):
        a_hat = np.append(a_hat,
                     alpha * (y[i] / F_fit[i - p]) + (1 - alpha) * 
                      (a_hat[i - 1] + b_hat[i -1]))
        b_hat = np.append(b_hat,
                     beta * (a_hat[i] - a_hat[i - 1]) + 
                      b_hat[i - 1] * (1 - beta))
        F_fit = np.append(F_fit, 
                 gamma * (y[i] / a_hat[i]) + F_fit[i - p] * (1 - gamma))
        y_hat = np.append(y_hat, 
                      (a_hat[i] + b_hat[i]) * F_fit[i])
    
    
    y_hat = y_hat.reshape(-1, 1)
    mape_train = np.sum(np.abs(y_hat - y)) / np.sum(y) * 100
    
    # Forecasting Model
    y_test = test.y.values.reshape(-1, 1)
    
    F=F_fit[-p:]
    a_hat_test=np.array([a_hat[-1]])
    b_hat_test=np.array([b_hat[-1]])
    y_hat_test = np.array([(a_hat_test + b_hat_test) * F[0]])
    
    for i in range(1, len(y_test)):
        a_hat_test = np.append(a_hat_test,
                     alpha * (y_test[i] / F[i - p]) + (1 - alpha) * 
                           (a_hat_test[i - 1] + b_hat_test[i - 1]))
        b_hat_test = np.append(b_hat_test,
                     beta * (a_hat_test[i] - a_hat_test[i - 1]) + 
                      b_hat_test[i - 1] * (1 - beta))
        F = np.append(F, 
                 gamma * (y_test[i] / a_hat_test[i]) + F[i - p] * (1 - gamma))
        y_hat_test = np.append(y_hat_test, 
                      (a_hat_test[i] + b_hat_test[i]) * F[i])
    
    y_hat_test = y_hat_test.reshape(-1, 1)
    
    sum_y=np.sum(y_test)
    sum_forecast=np.sum(y_hat_test)
    error=sum_y-sum_forecast
    percentage_error= 100 * np.sum(y_hat_test - y_test) / np.sum(y_test)
    mae=np.mean(np.abs(y_hat_test - y_test))
    root_error=np.sqrt(np.mean((y_hat_test - y_test) ** 2))
    mape = np.sum(np.abs(y_hat_test - y_test)) / np.sum(y_test) * 100
    
    print(sum_y)
    print(sum_forecast)
    print(error)
    print(percentage_error)
    print(mae)
    print(root_error)
    print(mape)
    
    plt.figure(figsize=(12,8))
    plt.plot(test['DATE'], y_hat_test, "--", label="Forecast")
    plt.plot(train['DATE'], y_hat, "--", label="Training Forecast")
    plt.plot(test['DATE'], test['y'], label="Actual")
    plt.plot(train['DATE'], train['y'], label="Training")
    plt.title("Holt Winters Forecast Test: MAPE = %.2f" %mape)
    plt.legend(loc="best")
    plt.show()
    

def main():
    holt_winter_model()
 
    
if __name__== "__main__":

    main()
