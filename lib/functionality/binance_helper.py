import pandas as pd
import numpy as np
import datetime as dt
import os
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor

# 
# def websocket_ticker(msg):
#     ''' define how to process incoming WebSocket messages '''
#     ta_rows = []
#     if msg['e'] != 'error':
#         global ta_df
#         ta_row = {}
#         ta_row.update(date = dt.datetime.now(),
#                       close = msg['c'])
#         ta_rows.append(ta_row)
#         ta_df = pd.DataFrame(ta_rows)
#         ta_df = ta_df.set_index('date')
#         ta_df['close'] = pd.to_numeric(ta_df.close)
        
#     else:
#         msg['error'] = True

def fetch_historic(params):
    symbol, frame, interval = [value for key, value in params.items()]
    """
    Returns historcal klines from past for given symbol and interval
    past_days: how many days back one wants to download the data
    """
    client = Client()
    if not interval:
        interval = '1h'
    if not frame:
        frame = 30

    start_str = str((pd.to_datetime('today') - pd.Timedelta(str(frame) + ' days')).date())
    
    data = pd.DataFrame(client.get_historical_klines(symbol = symbol,
                                                  start_str = start_str,
                                                  interval = interval))
    
    data.columns = [
        'open_time','open', 'high',
        'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades',
        'taker_base_vol', 'taker_quote_vol','is_best_match'
    ]
    
    data['open_date_time'] = [dt.datetime.fromtimestamp(x/1000) for x in data.open_time]
    data['symbol'] = symbol
    data = data[[
        'symbol','open_date_time','open', 
        'high', 'low', 'close', 'volume', 
        'num_trades', 'taker_base_vol', 'taker_quote_vol'
    ]]

    return data