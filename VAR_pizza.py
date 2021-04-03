"""
We followed the tutorial of Selva Prabhakaran of https://www.machinelearningplus.com
We made minor modfication to the model he created
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf

maxlag=12
test = 'ssr_chi2test'
var_1='PL_SUPRM'
var_3='DIG_3_MEAT'
var_4='DIG_SUPRM'
var_5='DIG_PEPP'
var_6='FRSC_BRCK'
var_7='FRSC_PEPP '
var_8='FRSC_4_CHEESE'
var_2='PL_MEAT'

def adjust(val, length= 6): 
    return str(val).ljust(length)
 
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))*100  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})
    
    
    
    
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df
            


def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)


#Augmented Dickey-Fuller Test (ADF Test)
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")    


def plot_timeseries(df):
     # Plot
    fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(10,6))
    for i, ax in enumerate(axes.flatten()):
        data = df[df.columns[i]]
        ax.plot(data, color='red', linewidth=1)
        # Decorations
        ax.set_title(df.columns[i])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)

        plt.tight_layout();
        
def plot_forecast(df,df_results,df_test,nobs):
    fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2, dpi=150, figsize=(10,10))
    for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
        df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
        df_test[col][-nobs:].plot(legend=True, ax=ax);
        ax.set_title(col + ": Forecast vs Actuals")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)

    plt.tight_layout();
    
        
def difference(df_train,df_test):
     ##################################################################################
    # 1st difference
    df_differenced = df_train.diff().dropna()
    
    # ADF Test on each column of 1st Differences Dataframe
    for name, column in df_differenced.iteritems():
        adfuller_test(column, name=column.name)
        print('\n')
    
    # Second Differencing
    df_differenced = df_differenced.diff().dropna()
    
    # ADF Test on each column of 2nd Differences Dataframe
    for name, column in df_differenced.iteritems():
        adfuller_test(column, name=column.name)
        print('\n')

    # ADF Test on each column
    # for name, column in df_train.iteritems():
    #    adfuller_test(column, name=column.name)
    #       print('\n')
    
    
    ##################################################################################
    return df_differenced
    
#serial correlation 
def durban_watson(model_fitted,df):
    out = durbin_watson(model_fitted.resid)
    def adjust(val, length= 6): return str(val).ljust(length)
    
    for col, val in zip(df.columns, out):
        print(adjust(col), ':', round(val, 2))
    
def invert_transformation(df_train, df_forecast, second_diff=True):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

   
def main():
    filepath='pizza.csv'
    df=pd.read_csv(filepath,parse_dates=['date'],index_col='date')
    print(df.shape)
    df.tail()
    
    #plot_timeseries(df)
    
    #cointegration_test(df)  
    
    nobs = 15
    df_train, df_test = df[0:-nobs], df[-nobs:]

    # Check size
    print(df_train.shape)
    print(df_test.shape)  
    
    df_differenced=difference(df_train,df_test)
   
    #grangers_causation_matrix(df, variables = df.columns) 
    
    model=VAR(df_differenced)
    for i in [1,2,3,4,5,6,7,8,9]:
        result=model.fit(i)
        print('Lag Order =',i)
        print('AIC :',result.aic)
        print('BIC :',result.bic)
        print('FPE :',result.fpe)
        print('HQIC :',result.hqic, '\n')
    
    #x = model.select_order(maxlags=12)
    #x.summary()
    
    model_fitted = model.fit(4)
    model_fitted.summary()
    
    durban_watson(model_fitted,df)
    
    # Get the lag order
    lag_order = model_fitted.k_ar
    print(lag_order)  #

    # Input data for forecasting
    forecast_input = df_differenced.values[-lag_order:]
    forecast_input
    
    # Forecast
    fc = model_fitted.forecast(y=forecast_input, steps=nobs)
    df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_2d')
    df_forecast
    
    df_results = invert_transformation(df_train, df_forecast, second_diff=True)        
    df_results.loc[:, ['{}_forecast'.format(var_1), '{}_forecast'.format(var_2), '{}_forecast'.format(var_3), '{}_forecast'.format(var_4),
                   '{}_forecast'.format(var_5), '{}_forecast'.format(var_6), '{}_forecast'.format(var_7), '{}_forecast'.format(var_8)]]
    
    plot_forecast(df,df_results,df_test,nobs)
   
    
    
    print('Forecast Accuracy of: {}'.format(var_1))
    accuracy_prod = forecast_accuracy(df_results['{}_forecast'.format(var_1)].values, df_test[var_1])
    for k, v in accuracy_prod.items():
        print(adjust(k), ': ', round(v,4))

    print('\nForecast Accuracy of: {}'.format(var_2))
    accuracy_prod = forecast_accuracy(df_results['{}_forecast'.format(var_2)].values, df_test[var_2])
    for k, v in accuracy_prod.items():
        print(adjust(k), ': ', round(v,4))

    print('\nForecast Accuracy of: {}'.format(var_3))
    accuracy_prod = forecast_accuracy(df_results['{}_forecast'.format(var_3)].values, df_test[var_3])
    for k, v in accuracy_prod.items():
        print(adjust(k), ': ', round(v,4))

    print('\nForecast Accuracy of: {}'.format(var_4))
    accuracy_prod = forecast_accuracy(df_results['{}_forecast'.format(var_4)].values, df_test[var_4])
    for k, v in accuracy_prod.items():
        print(adjust(k), ': ', round(v,4))

    print('\nForecast Accuracy of: {}'.format(var_5))
    accuracy_prod = forecast_accuracy(df_results['{}_forecast'.format(var_5)].values, df_test[var_5])
    for k, v in accuracy_prod.items():
        print(adjust(k), ': ', round(v,4))

    print('\nForecast Accuracy of: {}'.format(var_6))
    accuracy_prod = forecast_accuracy(df_results['{}_forecast'.format(var_6)].values, df_test[var_6])
    for k, v in accuracy_prod.items():
        print(adjust(k), ': ', round(v,4))

    print('\nForecast Accuracy of: {}'.format(var_7))
    accuracy_prod = forecast_accuracy(df_results['{}_forecast'.format(var_7)].values, df_test[var_7])
    for k, v in accuracy_prod.items():
        print(adjust(k), ': ', round(v,4))

    print('\nForecast Accuracy of: {}'.format(var_8))
    accuracy_prod = forecast_accuracy(df_results['{}_forecast'.format(var_8)].values, df_test[var_8])
    for k, v in accuracy_prod.items():
        print(adjust(k), ': ', round(v,4))


if __name__== "__main__":
    main()
