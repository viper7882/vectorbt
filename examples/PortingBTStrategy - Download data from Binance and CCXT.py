# pip install git+https://github.com/mementum/backtrader.git@0fa63ef4a35dc53cc7320813f8b15480c8f85517#egg=backtrader
# pip install -U python-binance
# pip install -U ccxt

import vectorbt as vbt
import backtrader as bt
from backtrader.sizers import PercentSizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import pytz

# Enter your parameters here
coin_target = "BTC"
coin_refer = "USDT"
symbol = '%s-%s' % (coin_target, coin_refer)
init_cash = 100
fees = 0.075 # in %
start_date = datetime(2021, 3, 10, tzinfo=pytz.utc)
end_date = datetime(2021, 3, 11, tzinfo=pytz.utc)
freq = '1m'

rsi_bottom = 35
rsi_top = 70
fast_window = 10
slow_window = 100

# ---------------------------------------------------------------------------------------
# Vectorbt Configurations
# https://vectorbt.dev/api/_settings/#vectorbt._settings.settings
# ---------------------------------------------------------------------------------------
vbt.settings.portfolio['freq'] = freq
vbt.settings.portfolio['init_cash'] = init_cash
vbt.settings.portfolio['fees'] = fees / 100
vbt.settings.portfolio['slippage'] = 0

# ---------------------------------------------------------------------------------------
# Pandas Configurations
# ---------------------------------------------------------------------------------------
# Display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Disable sequence of items (lists) to be truncated
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_seq_item', None)

# Python-Binance
binance_data = vbt.BinanceData.download(
    '%s%s' % (coin_target, coin_refer),
    start=start_date,
    end=end_date,
    interval='1m',
    tqdm_kwargs=dict(ncols='100%')
)
python_binance_data = binance_data.get()
python_binance_data.sort_index(inplace=True)
print(python_binance_data)

# CCXT
ccxt_data = vbt.CCXTData.download(
    '%s/%s' % (coin_target, coin_refer),
    start=start_date,
    end=end_date,
    timeframe='1m',
    tqdm_kwargs=dict(ncols='100%')
)
ccxt_binance_data = ccxt_data.get()
ccxt_binance_data.sort_index(inplace=True)
print(ccxt_binance_data)

data = python_binance_data

# Let's keep only the columns we're interested into
cols = ['Open', 'High', 'Low', 'Close', 'Volume']
ohlcv_wbuf = data[cols]

ohlcv_wbuf = ohlcv_wbuf.astype(np.float64)

print(ohlcv_wbuf.shape)
print(ohlcv_wbuf.columns)

wobuf_mask = (ohlcv_wbuf.index >= start_date) & (ohlcv_wbuf.index <= end_date) # mask without buffer
ohlcv = ohlcv_wbuf.loc[wobuf_mask, :]
print(ohlcv.shape)

# Plot the OHLC data
ohlcv_wbuf.vbt.ohlcv.plot().show()
