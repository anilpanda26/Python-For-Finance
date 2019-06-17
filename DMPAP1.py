import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime as dt
import matplotlib.pyplot as plt
from concurrent import futures as fts 
import os 

###################################################
# NOTES TO THE EXAMINER : Ideally one would wirte a method that can be re-used to calculate both Modified Faber Strategy and also 
# the SMA crossover, however I wasn't sure if that would satisfy the requirements of this assignment. 
# I have tried to address the individual questions to the best of my ability.Thank You, Anil Panda
###################################################

# Define the start and end date for the import 

#endDate = dt.now().date()
#startDate = endDate - pd.Timedelta(days = 365 * 10)
endDate = dt.date(2018, 12, 31)
startDate = dt.date(2013, 1, 1)

print (startDate, endDate)

# Import Microsoft Date from Yahoo Finance. 
MSFT = pdr.DataReader("MSFT", "yahoo", startDate, endDate)


# Calculate long term moviing averate 
vlma = 150

MSFTCopy = MSFT.copy()

MSFTCopy ['VLMA'] = MSFTCopy['Adj Close'].rolling(window = vlma, center = False).mean()

# Modified faber strategy. If the price is > VLMA the go long if price is < VLMA go short.

marketStatus ='Out'
buySignals = 0
sellSignals = 0
wealth = [1]

def modFaber (AdjClose, VLMA):
    global marketStatus
    global buySignals
    global sellSignals

    if AdjClose > VLMA and marketStatus == 'Out':
        marketStatus = 'In'
        buySignals = buySignals + 1 
        return 'Buy'

    if AdjClose < VLMA and marketStatus == 'In':
        marketStatus = 'Out'
        sellSignals = sellSignals + 1
        return 'Sell'

    else: 
        return 'Sit Tight'
    
MSFTCopy ['TS'] = MSFTCopy.apply(lambda x: modFaber(x['Adj Close'], x['VLMA']), axis = 1)

# Calculate the trade price, Buy/Sell Price. 
MSFTCopy['Trd Price'] = np.where( np.logical_or(MSFTCopy['TS'] == 'Buy', MSFTCopy['TS'] == 'Sell') , MSFTCopy['Adj Close'].shift(-1), 0)

print('BS', buySignals, 'SS', sellSignals)

# Get Buy and sell prices 
buyArr =[]
sellArr = []
def buySellPrices (TS , trdPrice):
    global buyArr
    global sellArr
    if TS == 'Buy':
        buyArr.append(trdPrice)

    if TS == 'Sell':
        sellArr.append(trdPrice)

MSFTCopy.apply(lambda x: buySellPrices (x['TS'], x['Trd Price']), axis = 1)


# Calculate Wealth using cumulative prod value. 
def trdRtn (buyArray, sellArray, wealthArr):
    
    for d in range(len(buyArray)):
        wealthArr.append(sellArr[d]/buyArray[d])
    wealthArr= np.cumprod(wealthArr)

trdRtn (buyArr, sellArr, wealth)

#******************************************************************************** 
# Writign an independent function to calculate the and compare the return for modified Faber stragety

def resetGlobalVars ():
    global marketStatus
    global buySignals
    global sellSignals
    global wealth
    global buyArr
    global sellArr
    global wealth
    marketStatus = 'Out'
    buySignals = 0
    sellSignals = 0
    wealth = [1]
    buyArr =[]
    sellArr = []
    wealth = [1]

VLSMA = [150, 160, 170, 180, 190, 200]
maWealth = []
MSFTCP = MSFT.copy()

def calWealth (maVal):
    MSFTCP['VLSMA' + str(maVal)] = MSFTCP['Adj Close'].rolling(window = maVal, center = False).mean()
    
   # Apply Modified faber value to calculate signal
    MSFTCP['TSig' + str(maVal)] = MSFTCP.apply(lambda x: modFaber( x['Adj Close'], x['VLSMA' + str(maVal)]), axis = 1)
  
   # Calculate trade Prices 
    MSFTCP['TP' + str(maVal) ] = np.where( np.logical_or(MSFTCP['TSig' + str(maVal)] == 'Buy', MSFTCP['TSig' + str(maVal)] == 'Sell') , MSFTCP['Adj Close'].shift(-1), 0)
    
    # Create Array of Buy and Sell Prices. 
    MSFTCP.apply(lambda x: buySellPrices (x['TSig' + str(maVal)], x['TP' + str(maVal)]), axis = 1)
    
    # Caculate  Cumulative Wealth.
    trdRtn (buyArr, sellArr, wealth)
    return wealth
    

# calculate wealth program for multiple periods
for x in VLSMA:
    resetGlobalVars ()
    maWealth.append(calWealth(x))
wltReturns = pd.DataFrame(data = maWealth)
wltReturns = wltReturns.transpose()

plt.figure(figsize=(14,9))
plt.grid(True)
plt.title('Wealth Returns')
plt.plot(wltReturns[0], linewidth= 1, label = 'MA150') 
plt.plot(wltReturns[1], linewidth= 1, label = 'MA160')
plt.plot(wltReturns[2], linewidth= 1, label = 'MA170')
plt.plot(wltReturns[3], linewidth= 1, label = 'MA180')
plt.plot(wltReturns[4], linewidth= 1, label = 'MA190')
plt.plot(wltReturns[5], linewidth= 1, label = 'MA200')
plt.legend()
plt.show()

################################################
# Moving Average Crossover strategy 
################################################

SMA150 = 150
SMA20 = 20

MSFTCX = MSFT.copy()
# Calculate Moving Average
MSFTCX['SMA150'] = MSFTCX['Adj Close'].rolling(window = 150, center = False).mean()
MSFTCX['SMA50'] = MSFTCX['Adj Close'].rolling(window = 50, center = False).mean()

# Calculate Moving Previous Day
MSFTCX['SMA150Prv'] = MSFTCX['Adj Close'].rolling(window = 150, center = False).mean().shift(1)
MSFTCX['SMA50Prv'] = MSFTCX['Adj Close'].rolling(window = 50, center = False).mean().shift(1)

# Calculate Moving Averages Cross-over 
MSFTCX['Sig'] = np.where((MSFTCX['SMA50'] > MSFTCX['SMA150']) & (MSFTCX['SMA50Prv'] < MSFTCX['SMA150Prv']), 1, 0)
MSFTCX['Sig'] = np.where((MSFTCX['SMA50'] < MSFTCX['SMA150']) & (MSFTCX['SMA50Prv'] > MSFTCX['SMA150Prv']), -1, MSFTCX['Sig'])

# Calculate Position 
MSFTCX['Position'] = MSFTCX['Sig'].replace (to_replace = 0, method = 'ffill')

# Calculate Trade Prices 
MSFTCX['TP'] = np.where (np.logical_or(MSFTCX['Sig']== 1, MSFTCX['Sig']== -1), MSFTCX['Adj Close'], 0)
MSFTCX['TP'] = MSFTCX['TP'].replace (to_replace = 0, method = 'ffill')

# Calculate Returns 
MSFTCX['Buy & Hold Returns'] = np.log(MSFTCX['Adj Close'] / MSFTCX['Adj Close'].shift(1))
MSFTCX['Strategy Returns'] = MSFTCX['Buy & Hold Returns'] * MSFTCX['Position'].shift(1)

# Plot Cumulative Returns
MSFTCX[['SMA150', 'SMA50']].plot(grid=True, figsize=(9,5))
MSFTCX[['Buy & Hold Returns', 'Strategy Returns']].cumsum().plot(grid=True, figsize=(9,5))




print (MSFTCX['Sig'].value_counts())
print(MSFTCX['Position'].tail())


plt.figure(figsize=(10, 5))
plt.plot(MSFTCX['Position'])
plt.title("Signal buy/sell positions")
plt.xlabel('Time')
plt.tight_layout()
plt.show()

##########################################
#  Downloand multiple concurrent stocks 
##########################################

stockList = ['TSLA', 'NFLX', 'AMZN', 'GOOG', 'CVX']
downLoadedStocks = []
problemStocks = []

# Function definition to handle download 

def downLoadStock (stock):
    try:
        print('Trying to retrieve the %s ... \n' %(stock))
        
        stockDf = pdr.DataReader(stock, 'yahoo', startDate, endDate)
        stockDf['Name'] = stock
        outputName = stock + '.csv'
        downLoadedStocks.append(outputName)
        stockDf.to_csv(outputName)
        print('%s download complete \n' %(stock))
    except:
        problemStocks.append(stock)
        print('%s could not be downloaded \n' %(stock))

# Define number of workers 

# maxWorkers = 20
# ttlWorkers = min(maxWorkers, len(stockList))
# print(ttlWorkers)
# print (os.getcwd())
# with fts.ThreadPoolExecutor(ttlWorkers) as executors:
#     res = executors.map(downLoadStock, stockList)