from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor

import pandas as pd
import numpy as np
import datetime as dt
import os


def websocket_ticker(msg):
    """
    Retrieves current live price data from a binance websocket\n
        * :param msg(json): Current asset price\n
    
    :return data(pd.Series): Pandas series containing received price data
    """
    rows = []
    if msg['e'] != 'error':
        row = {}
        row.update(
            date = dt.datetime.now(),
            close = msg['c']
        )
        rows.append(row)
        df = pd.DataFrame(rows).set_index('date')
        df['close'] = pd.to_numeric(df.close)
        return df
        
    else:
        msg['error'] = True

def fetch_historic(params):
    """
    Retrieves OHLC data from the binance API\n
        * :param params(dict): Dictionary of parameters relating to time frame, interval and ticker\n
    
    :return data(pd.DataFrame): Dataframe of OHLC+ data for supplied parameters
    """
    client = Client()
    symbol, frame, interval = [value for key, value in params.items()]
    
    if not interval: interval = '1h'
    if not frame: frame = 30

    start_str = str(
        (pd.to_datetime('today') - pd.Timedelta(str(frame) + ' days')
    ).date())
    
    data = pd.DataFrame(
        client.get_historical_klines(
            symbol = symbol,
            start_str = start_str,
            interval = interval
        )
    )
    
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