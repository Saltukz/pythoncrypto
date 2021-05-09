import pandas as pd

import datetime as dt
from datetime import datetime, timedelta

import pytz

import time
import dateparser

from binance.client import Client

from binance.client import Client

from financialQueries import *



#Construct Binance Client

api_key =''
api_secret = ''
client = Client(api_key,api_secret)

#Get Market Depth

depth = client.get_order_book(symbol = 'ANTUSDT')


## Define binance helper functions

def binanceDataFrame(klines):
  df = pd.DataFrame(klines.reshape(-1,12),dtype=float, columns=['Open Time',
                                                                 'Open',
                                                                 'High',
                                                                 'Low',
                                                                 'Close',
                                                                 'Volume',
                                                                 'Close Time',
                                                                 'Quote asset volume',
                                                                 'Number of trades',
                                                                 'Taker buy base asset volume',
                                                                 'Taker buy quote asset volume',
                                                                 'Can be ignored']);
  df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
  df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
  return df


def date_to_miliseconds(date_str):
  epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
  d = dateparser.parse(date_str)
  if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
    d = d.replace(tzinfo=pytz.utc)

     #return difference in time
  return int((d - epoch).total_seconds()*1000.0)


def interval_to_miliseconds(interval):
  ms = None
  seconds_per_unit = {
    "m":60,
    "h":60*60,
    "d":24 * 60 * 60,
    "w":7*24*60*60,
  }
  unit = interval[-1]
  if unit in seconds_per_unit:
    try:
      ms=int(interval[:-1]) * seconds_per_unit[unit] * 1000
    except:
      pass
  
  return ms

def get_historical_klines(symbol, interval, start_str, end_str=None):
  output_data=[]
  limit=500
  timeFrame = interval_to_miliseconds(interval)
  start_ts = date_to_miliseconds(start_str)
  end_ts = None
  if end_str:
    end_ts = date_to_miliseconds(end_str)
  
  idx = 0
  symbol_existed = False
  while True:
    #fetch
    temp_data = client.get_klines(symbol = symbol, 
                                  interval = interval, 
                                  limit = limit, 
                                  startTime = start_ts, 
                                  endTime = end_ts)
    
    if not symbol_existed and len(temp_data):
      symbol_existed = True

    if symbol_existed:
      output_data += temp_data

      start_ts = temp_data[len(temp_data) - 1][0] + timeFrame

    idx += 1

    if len(temp_data)<limit:
      break

    if idx % 3 == 0:
      time.sleep(1)

  return output_data
