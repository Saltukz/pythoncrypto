import pandas as pd

import datetime as dt
from datetime import datetime, timedelta

import pytz

import time
import dateparser

import requests



from financialQueries import *



def gateioDataFrame(r):
  df = pd.DataFrame(r.reshape(-1,6),dtype=float, columns=['Open Time',
                                                                 'Volume',
                                                                 'Close',
                                                                 'High',
                                                                 'Low',
                                                                 'Open']);
  df['Open Time'] = pd.to_datetime(df['Open Time'], unit='s')
 
  return df



