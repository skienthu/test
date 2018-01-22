
# coding: utf-8

# In[431]:

import pandas as pd
import datetime
import numpy as np
from dateutil.relativedelta import relativedelta
import pandas_datareader as pdr
from pandas_datareader._utils import RemoteDataError
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import pylab

#read the csv file and index with date

##Some nice code to convert daily to weekly but capture min max and volume for entire week

def take_first(array_like):
    return array_like[0]

def take_last(array_like):
    return array_like[-1]

def take_average(array_like):
    return array_like.mean()

def d2w(df, mode='W'):
    mDays = {"W":-4,"M":-31}
    output = df.resample(mode, how={'Open': take_first,
                                    #'Avergae': 'mean',
                                     'High': 'max',
                                     'Low': 'min',
                                     'Close': take_last,
                                     'Volume': 'sum'}, 
                                    loffset=pd.offsets.timedelta(days=mDays[mode]))

    output = output[['Open','High', 'Low', 'Close', 'Volume']]
    
    return output


def changeDate(str, format="%d/%m/%Y"): return datetime.datetime.strptime(str,format).date()

def readCSV(csvPath):
    data = pd.read_csv(csvPath)
    data.index = pd.to_datetime(data["Date"].apply(changeDate))
    del data['Date']
    return data.sort_index(ascending = True)

#Rolling window
def getZscore(returns,lookBack):
    
    lRtn_mean = returns.rolling(lookBack).mean().dropna()
    lRtn_std = returns.rolling(lookBack).std().dropna()
    zScore = ((returns - lRtn_mean).dropna() / lRtn_std).dropna()
    return zScore

#expanding window

def getExpZscore(returns, minWindow=22):
    
    lRtn_mean = returns.expanding(minWindow).mean().dropna()
    lRtn_std = returns.expanding(minWindow).std().dropna()
    zScore = ((returns - lRtn_mean).dropna() / lRtn_std).dropna()
    return zScore

def getEWMAVector( wts, rFactor=1):
    eWts = [np.exp(i) for i in wts]
    eWts = [ i / np.sum(eWts) for i in eWts]
    eWts = pd.Series(eWts)
    eWts = pd.concat([eWts]* rFactor ,ignore_index = True)
    return eWts
    
def ewma(df, days=5):
    length = len(mResult)
    lB = uB = 0
    offSet = days -1
    uB = lB + offSet
    indexes = df.index
    weeklyMVA = pd.DataFrame()
    while(uB < length):
        lBound = indexes[lB]
        uBound = indexes[uB]
        x = pd.DataFrame(df[lBound:uBound].sum(axis=0)).T
        x.index = [uBound]
        weeklyMVA = pd.concat([weeklyMVA,x],axis=0)
        lB = uB
        uB = uB + offSet
    
    return weeklyMVA.sort_index()

def getAverageWeeklyPrices(dSource):
    nFriday = [x.to_datetime().date() + relativedelta(days = 4 - x.to_datetime().date().weekday()) for x in dSource.index]
    dSource1 = pd.concat([pd.DataFrame(nFriday,columns = ['Date'],index= dSource.index),
                          dSource],axis=1).groupby(['Date']).mean().reset_index()

    dSource1.index = pd.to_datetime(dSource1['Date'])
    del dSource1['Date']
    
    return dSource1

def getAverageMonthlyPrices(dSource):
    nFriday = [x.to_datetime().date() + relativedelta(day = 31) for x in dSource.index]
    dSource1 = pd.concat([pd.DataFrame(nFriday,columns = ['Date'],index= dSource.index),
                          dSource],axis=1).groupby(['Date']).mean().reset_index()

    dSource1.index = pd.to_datetime(dSource1['Date'])
    del dSource1['Date']
    
    return dSource1

def getDataFromYahoo(symbol, startDate, endDate):
    prices = pd.DataFrame()
    ctr = 0;
    while ctr < 5:
        try:
            prices = pdr.get_data_yahoo(symbols=symbol, start=startDate, end=endDate)
            break;
            
        except RemoteDataError:
            ctr = ctr + 1
            print("Lode lag gaye attempt .."+ str(ctr) +  " - " + symbol )
            
    return prices

def movingAverage(df,window=200):
    length = len(df)
    lB = uB = 0
    offSet = window -1
    uB = lB + offSet
    indexes = df.index
    weeklyMVA = pd.DataFrame()
    while(uB < length):
        lBound = indexes[lB]
        uBound = indexes[uB]
        x = pd.DataFrame(df[lBound:uBound].mean(axis=0)).T
        x.index = [uBound]
        weeklyMVA = pd.concat([weeklyMVA,x],axis=0)
        lB = uB
        uB = uB + offSet
    
    return weeklyMVA.sort_index()


def regress(Y,X):
    est = sm.OLS(Y,X)
    est2 = est.fit()
    return est2

def getRegressionParams(Y,X):
    allParams = pd.DataFrame()
    est2 = regress(Y,X)
    return pd.DataFrame(est2.params).T

def rollingRegression(X,Y, lookBack, window = 'r'):
    cDates = sorted(set(X.index).intersection(set(Y.index)))
    X = X.loc[cDates]
    Y = Y.loc[cDates]
    lB = uB = 0
    offSet = lookBack - 1
    lDate = cDates[lB]
    uDate = cDates[uB]
    rParams = pd.DataFrame()
    while(uB < len(cDates)):
        x = X[cDates[lB]:cDates[uB]]
        y = Y[cDates[lB]:cDates[uB]]
        rParam = getRegressionParams(y,x)
        rParam.index = [cDates[uB].date()]
        rParams = pd.concat([rParams, rParam], axis=0)
        if(window == "r" ):
            lB = lB+1
            uB = uB+1
        else:
            uB = uB+1

    return rParams


# In[432]:

commonDates = sorted(set(dailyData.index).intersection(set(dailyData.index)))
X = dailyData.loc[commonDates]
Y = dailyData.loc[commonDates]
print (rollingRegression(X['SPX Index'],Y['SPX Index'],100,"e"))


# In[ ]:




# In[ ]:

dailyData =  readCSV("Data/spx.csv")
dailyData =  dailyData[:-(len(dailyData)%5)]
# "sample" function to sample random data from the dataframe
daily_Returns = np.log(dailyData.sort_index()).diff().dropna()
zScoreRolling = getZscore(daily_Returns,22)
zScoreExpanding = getExpZscore(daily_Returns,22)

weeklyData = dailyData.resample("W-MON", how='last')
monthlyData = dailyData.resample("M").first()

x = [1,2,3,4,5]
numWeeks = int(len(dailyData)/5)
ewmaVector = getEWMAVector(x,numWeeks)
ewmaVector.index = dailyData.index

mResult = (dailyData.mul(ewmaVector, axis=0).sort_index(ascending=True))
(dailyData.rolling(200).mean().dropna() - dailyData.rolling(5).mean().dropna()).dropna()

'''
startDate = datetime.datetime(1990, 1, 1) 
endDate = datetime.datetime(2018, 1, 12)
x_snp = pd.DataFrame(getDataFromYahoo("^GSPC", startDate, endDate))
x_snp.to_csv("test.csv")
x_snp.index = pd.to_datetime(x_snp.index)
x_snp_w = d2w(x_snp,'W')
'''


# In[ ]:




# In[ ]:




# In[ ]:



