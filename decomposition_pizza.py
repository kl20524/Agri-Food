"""
We followed the tutorial by C hubbs on the datahubbs blog
https://www.datahubbs.com/machine-learning-supply-chain-forecasting-2/ 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

url = 'pizza_total.csv'


def yearly_cycle(test,train):
    df_year=pd.DataFrame({"y":train.y.resample("M").sum()})#resample monthly
    df_year['mva']=df_year.y.rolling(center=True,window=12).mean()#moving average
    
    #plt.figure(figsize=(12,8))
    #plt.plot(df_year['y'], label='Actual')
    #plt.plot(df_year['mva'], label='Monthly Average')
    #plt.legend(loc='best')
    #plt.show()
    
    df_year['sr'] = df_year['y'] / df_year['mva']
    
    # Add month numbers
    df_year['month'] = df_year.index.month
    
    # Average each month 
    df_ = df_year.groupby('month').agg({
        'sr': 'mean'})
    df_.reset_index(inplace=True)
    df_.columns = ['month', 'usi']
    
    # Combine with main data frame
    df_year = pd.merge(df_year, df_, on='month', right_index=True).sort_index()
    
    df_year['asi'] = df_['usi'].mean() * df_year['usi']#Adjusted Seasonal Index
    df_year['y_t-s'] = df_year['y'] / df_year['asi']#e-seasonalized values (yt−s):
    
    #plt.figure(figsize=(12,8))
    #plt.plot(df_year['y'], label='$y$')
    #plt.plot(df_year['y_t-s'], label='$y_{t-s}$')
    #plt.legend(loc='best')
    #plt.show()
    
    lm = LinearRegression(normalize=False, fit_intercept=True)
    y_t_s = np.atleast_2d(df_year['y_t-s'].values).T
    x = np.atleast_2d(np.linspace(0, len(df_year) - 1, len(df_year))).T
    lm.fit(x, y_t_s)
    df_year['trend'] = lm.predict(x)#adds trends
    
    # Plot actual data, de-seasonalized data, and the trend
    #plt.figure(figsize=(12,8))
    #plt.plot(df_year['y'], label='$y$')
    #plt.plot(df_year['y_t-s'], label='$y_{t-s}$')
    #plt.plot(df_year['trend'], label="$T'$")
    #plt.legend(loc='best')
    #plt.show()
    
    df_year['noise']=(df_year['y']/(df_year['asi']*df_year['trend'])).mean()
    df_year.head()
    
    test_year=pd.DataFrame({"y": test.y.resample("M").sum()})
    test_year['month']=test_year.index.month
    
    #get index for trend regression
    x_test = np.linspace(len(df_year), len(df_year) + 
                     len(test_year) - 1,
                    len(test_year)).reshape(-1,1)
    df_test = pd.merge(test_year, df_year[['month', 'asi', 'noise']],
                       on='month', 
                       right_index=True).sort_index().drop_duplicates()
    df_test['trend'] = lm.predict(x_test)
    df_test['forecast'] = df_test['asi'] * df_test['noise'] * df_test['trend']
    df_test
    
    plt.figure(figsize=(12,8))
    plt.plot(df_year['y'], label='Train ($y_t$)')
    plt.plot(df_test['y'], label='Test ($y_t$)')
    plt.plot(df_test['forecast'], label='Forecast ($\hat{y_t}$)')
    plt.legend(loc='best')
    plt.title("Classical Decomposition and Multiplicative Model Forecast")
    plt.show()
    
    evaluation = df_test.copy()
    evaluation['error'] = evaluation['y'] - evaluation['forecast']
    evaluation.insert(0, 'series', 1) # insert value to groupby
    df=evaluation.groupby('series').agg({
        'y' : 'sum',
        'forecast' : 'sum',
        'error': {
            'total_error' : 'sum',
            'percentage_error' : lambda x: 100 * np.sum(x) / np.sum(evaluation['y']),
            'mae': lambda x: np.mean(np.abs(x)),
            'rmse': lambda x: np.sqrt(np.mean(x ** 2)),
            'mape': lambda x: 100 * np.sum(np.abs(x)) / np.sum(evaluation['y'])
        }}).apply(np.round, axis=1)
    
    print ("\n")
    print (df)
    print ("\n")


def main():
    
    data=pd.read_csv(url,sep=',')
    data.columns=['DATE','y']
    #drop removes data
    data=data.set_index(pd.to_datetime(data.ix[:,0])).drop('DATE',axis=1)
    data.head()
    train= data.loc[data.index<'1/12/2011']
    test= data.loc[data.index>'1/12/2011']
    yearly_cycle(test,train)
    
    
if __name__== "__main__":
    main()
