import pandas as pd
import numpy as np


import time
import dateparser
import pytz
import json

import math  
import requests
import datetime

import dateutil.parser


from threading import Thread



import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.dates as mpl_dates

from mplfinance.original_flavor import candlestick_ohlc

from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv

from tqdm import tqdm as tqdm

from binance.client import Client

from financialQueries import *

from binanceHelpers import *

from gateioHelpers import *

from Coin import *

import gate_api

import requests

import finplot as fplt


#GET THE DATA
content = requests.get("https://api.binance.com/api/v3/exchangeInfo").json()
content = content.get("symbols")
symbols = ['BTCUSDT']
# for thing in content:
#     if thing.get("status") == "TRADING" and thing.get("isSpotTradingAllowed") == True:
#         symbols.append(thing.get("symbol"))

print(symbols)
for pair in symbols:
  symbol = pair
  start = "1 Jan, 2021"
  interval = Client.KLINE_INTERVAL_4HOUR
  klines = get_historical_klines(symbol,interval,start)
  klines = np.array(klines)
  df = binanceDataFrame(klines)


  # remove_cols = [c for c in df.columns if c not in ['Open','Close','High','Low']]
  # df.drop(remove_cols, axis=1, inplace=True)

  #Calculate financial indicators for several windows

  windows = [6]
  windowsForMa = [7,25,99]

  for w in tqdm(windowsForMa):
    df = moving_average(df,w)
    df = exponential_moving_average(df,w)


  for w in tqdm(windows):

    df = momentum(df,w);
    f = bollinger_bands(df,w);
    df = stochastic_oscillator(df,w);
    df = RSI(df,w);
    df = common_channel_index(df,w);
    df = standart_deviation(df,w);
    print(df)
    
    
 
# create two axes
ax,ax2 = fplt.create_plot(symbol, rows=2)

# plot candle sticks
candles = df[['Open Time','Open','Close','High','Low']]
fplt.candlestick_ochl(candles, ax=ax)

# overlay volume on the top plot
volumes = df[['Open Time','Open','Close','Volume']]
fplt.volume_ocv(volumes, ax=ax.overlay())

# put an MA on the close price
fplt.plot(df['Open Time'], df['Close'].rolling(25).mean(), ax=ax, legend='ma-25')
fplt.plot(df['Open Time'], df['MA_99'], ax=ax, legend='ma-99')

# place some dumb markers on low wicks
lo_wicks = df[['Open','Close']].T.min() - df['Low']
df.loc[(lo_wicks>lo_wicks.quantile(0.99)), 'marker'] = df['Low']
fplt.plot(df['Open Time'], df['marker'], ax=ax, color='#4a5', style='^', legend='dumb mark')

# draw some random crap on our second plot
fplt.plot(df['Open Time'], np.random.normal(size=len(df)), ax=ax2, color='#927', legend='stuff')
fplt.set_y_range(-1.4, +3.7, ax=ax2) # hard-code y-axis range limitation

# restore view (X-position and zoom) if we ever run this example again
fplt.autoviewrestore()

# we're done
#fplt.show()   
    











  ##############################################
# plt.figure(figsize=(10,10))

# plt.plot(df['Close Time'],df['MA_7'], 'g--', label="MA_7")
# plt.plot(df['Close Time'],df['MA_25'], 'r--', label="MA_25")
# plt.plot(df['Close Time'],df['MA_99'], 'b--', label="MA_99")
# plt.plot(df['Close Time'], df['Close'])
# plt.legend()
# plt.xlabel("date")
# plt.ylabel("$ price")
# plt.title(pair)

# plt.show()
  
# for obj in locals().values():
#   print(obj)
###############################################

####################################################################
# trades = client.get_recent_trades(symbol='BTCUSDT')

# print(trades)
###################################################################

## gate.io



hostgate = "https://api.gateio.ws"
prefixgate = "/api/v4"
headersgate = {'Accept': 'application/json', 'Content-Type': 'application/json'}

urlgate = '/spot/candlesticks'
query_paramgate = 'currency_pair=BTC_USDT'
intervalgate = 'interval=4h'
limitgate = 'limit=740'

sgate = "1 Jan, 2021"
# timegate = time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple())

timegate = date_to_miliseconds(sgate)/1000
timegateint = int(timegate)
timegatestr= str(timegateint)

querytimegate = 'from=' + timegatestr

print(querytimegate)
r = requests.request('GET', hostgate + prefixgate + urlgate + "?" + query_paramgate + "&" + intervalgate + "&" + querytimegate, headers=headersgate)
r= np.array(r.json())


c = gateioDataFrame(r)

print(c)


