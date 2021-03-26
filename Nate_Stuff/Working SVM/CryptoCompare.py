# testing out the Crypto Compare API

# We may run out of api calls

import cryptocompare as cc
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

KEY = "b14d953abbd87596c1c1c75e5c2318eea939bc1ddb17a113355b8a4cca312488"
cc.cryptocompare._set_api_key_parameter(KEY)

# Minute data is only available for the past 7 days
# Takes about 2.5s to load
def get_cc_minute_prices(coin, days=7):
    # This will return a list of dicts! Whoopyyyy
    day_stamps = [datetime.now() - timedelta(hours=24 * i) for i in range(days)]
    df = pd.DataFrame(cc.get_historical_price_minute(coin, 'USDT', exchange='Binance', toTs = day_stamps[0]))

    for j in range(1, days):
        old_day = pd.DataFrame(cc.get_historical_price_minute(coin, 'USDT', exchange='Binance', toTs = day_stamps[j]))
        df = pd.concat([old_day, df])

    times = df['time'].tolist()
    newtimes = []
    for timeval in times:
        newtimes.append(datetime.utcfromtimestamp(timeval).strftime('%Y-%m-%d %H:%M:%S'))
    df['time'] = newtimes
    df.reset_index(drop=True, inplace=True)
    return df

# Hour data is available for a very long time
# Takes a fair while to load but who cares when you have an hour for each prediction
# I know for a fact that you can get 5 years of data in roughly 15 seconds
def get_cc_hour_prices(coin, calls=15):
    # This will return a list of dicts! Whoopyyyy
    call_stamps = [datetime.now() - timedelta(hours=1440 * i) for i in range(calls)]
    df = pd.DataFrame(cc.get_historical_price_hour(coin, 'BUSD', exchange='Binance', toTs = call_stamps[0]))

    for j in range(1, calls):
        old_call = pd.DataFrame(cc.get_historical_price_hour(coin, 'BUSD', exchange='Binance', toTs = call_stamps[j]))
        df = pd.concat([old_call, df])

    times = df['time'].tolist()
    newtimes = []
    for timeval in times:
        newtimes.append(datetime.utcfromtimestamp(timeval).strftime('%Y-%m-%d %H:%M:%S'))
    df['time'] = newtimes
    df.reset_index(drop=True, inplace=True)
    return df

@logger.catch
def main():
    # now = datetime.now()
    # df = get_cc_minute_prices('BTC')
    # print(df[['time', 'open', 'high', 'low', 'close']])
    # print('Download time: ', datetime.now()-now)

    now = datetime.now()
    df = get_cc_hour_prices('BTC')
    print(df[['time', 'open', 'high', 'low', 'close']])
    print(df['volumefrom'].tail(60))
    print('Download time: ', datetime.now()-now)


if __name__ == '__main__':
    main()
