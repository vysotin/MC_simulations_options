import os
import pathlib
from pathlib import Path
import pandas as pd
import sys
import csv
from pprint import pprint
import datetime
from dotenv import load_dotenv

from utils import str_to_epoch

load_dotenv()

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(''))))
sys.path.append(root + '/python')

CURRENT_FILE_DIR = pathlib.Path(__file__).parent.resolve()

import ccxt



def get_exchange(exchange_id):
    exchange_class = getattr(ccxt, exchange_id)
    api_key = os.environ.get(exchange_id.upper() + '_API_KEY')
    api_secret = os.environ.get(exchange_id.upper() + '_API_SECRET')
    exchange = exchange_class({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True
    })
    return exchange


def fetch_ohlcv_with_retries(exchange,symbol, timeframe, since, limit=20000, max_retries=3):
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        return ohlcv
    except Exception as ex:
        print(ex)
        if num_retries > max_retries:
            raise  # Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')


def load_ohlcv(exchange, symbol, timeframe, since = '2025-01-01', limit=20000, max_retries=3):
    earliest_timestamp = exchange.milliseconds()
    timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    timedelta = limit * timeframe_duration_in_ms
    all_ohlcv = []
    if isinstance(since, str):
        since = str_to_epoch(since)*1000

    while True:
        fetch_since = earliest_timestamp - timedelta
        if fetch_since < since:
            fetch_since = since
        ohlcv = fetch_ohlcv_with_retries(exchange, symbol, timeframe, fetch_since, max_retries=max_retries, limit=limit)
        # if we have reached the beginning of history
        if not ohlcv:
            break
        if ohlcv[0][0] >= earliest_timestamp:
            break
        earliest_timestamp = ohlcv[0][0]
        all_ohlcv = ohlcv + all_ohlcv
        print(len(all_ohlcv), symbol, 'candles in total from', exchange.iso8601(all_ohlcv[0][0]), 'to', exchange.iso8601(all_ohlcv[-1][0]))
        # if we have reached the checkpoint
        if earliest_timestamp <= since:
            break
    return all_ohlcv


def write_to_csv(dirpath: Path, filename, data):
    if isinstance(dirpath, str):
        dirpath = pathlib.Path(dirpath)
    full_path = dirpath/str(filename)
    with Path(full_path).open('w+', newline='') as output_file:
        csv_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(data)


def load_prices_to_csv(dirpath, filename, exchange, symbol, timeframe, since, limit=20000, max_retries=3):
    # instantiate the exchange by id
    # convert since from string to milliseconds integer if needed
    if isinstance(since, str):
        since = str_to_epoch(since)*1000
    # preload all markets from the exchange
    exchange.load_markets()
    # fetch all candles
    ohlcv = load_ohlcv(exchange, symbol, timeframe, since, max_retries=max_retries, limit=limit)
    # save them to csv file
    if ohlcv:
        write_to_csv(dirpath, filename, ohlcv)
        print('Saved', len(ohlcv), 'candles from', exchange.iso8601(ohlcv[0][0]), 'to', exchange.iso8601(ohlcv[-1][0]),
              'to', f"{dirpath}/{filename}")
    else:
        print('No candles found for', symbol, 'from', exchange.iso8601(since), 'to',
              exchange.iso8601(since + limit * exchange.parse_timeframe(timeframe)))


def load_all_option_metadata_for_symbol(exchange, underlying_symbol):
    exchange.options["fetchMarkets"] = ['option']
    exchange.load_markets()
    all_markets = exchange.fetch_markets(params={'type': 'option', 'limit': 100000})

    # Filter for option contracts with the specified underlying
    option_contracts = {}
    for metadata in all_markets:
        if metadata['base'] == underlying_symbol:
            option_contracts[metadata['symbol']] = metadata
    return option_contracts


def get_today_datelabel():
    now = datetime.datetime.now()
    return now.strftime("%y%m%d")

def download_option_prices(exchange, underlying_symbol, base_curr, nom_currency, timeframe, expiry_date_label,
                            since_date='2024-01-01 00:00:00Z', upto_date=None, root_dir=CURRENT_FILE_DIR):
    if not upto_date:
        upto_date = get_today_datelabel()
    p_call = get_dir_option_price_data_path(exchange, underlying_symbol, timeframe, upto_date, expiry_date_label,
                                            'call')
    p_call.mkdir(parents=True, exist_ok=True)
    p_put = get_dir_option_price_data_path(exchange, underlying_symbol, timeframe, upto_date, expiry_date_label,
                                           'put')
    p_put.mkdir(parents=True, exist_ok=True)
    option_contracts = load_all_option_metadata_for_symbol(exchange, underlying_symbol)
    for market, metadata in option_contracts.items():
        if market.startswith(f"{underlying_symbol}/{base_curr}:{nom_currency}-{expiry_date_label}"):
            filename = market.replace("/", "_").replace(f":", "_") + ".csv"
            if metadata['optionType'] == 'call':
                load_prices_to_csv(p_call, filename, exchange=exchange,
                                   symbol=market, timeframe=timeframe, since=since_date)
            else:
                load_prices_to_csv(p_put, filename, exchange=exchange,
                                   symbol=market, timeframe=timeframe, since=since_date)

def download_spot_prices(exchange, underlying_symbol, base_curr, nom_currency, timeframe,
                         since_date='2024-01-01 00:00:00Z', upto_date=None):
    root_dir = get_dir_spot_price_data_path(exchange, underlying_symbol, upto_date=upto_date, timeframe=timeframe)
    root_dir.mkdir(parents=True, exist_ok=True)
    symbol = f"{underlying_symbol}/{base_curr}:{nom_currency}"
    filename = symbol.replace("/", "_").replace(f":", "_") + ".csv"
    load_prices_to_csv(root_dir, filename, exchange=exchange,
                       symbol=symbol, timeframe=timeframe, since=since_date)

def download_option_and_spot_prices(exchange_name, underlying_symbol, timeframe, expiry_date_label,
                                     since_date='2024-01-01 00:00:00Z', upto_date=None):

    exchange = get_exchange(exchange_name)
    base_curr = 'USD' if exchange_name == 'deribit' else 'USDT'
    nom_currency = underlying_symbol if exchange_name == 'deribit' else 'USDT'
    upto_date_label = get_today_datelabel() if upto_date is None else upto_date
    download_option_prices(exchange, underlying_symbol, base_curr, nom_currency, timeframe, expiry_date_label, since_date,
                           upto_date=upto_date_label)
    download_spot_prices(exchange, underlying_symbol, base_curr, nom_currency, timeframe, since_date,
                         upto_date=upto_date_label)

def get_dir_option_price_data_path(exchange, underlying_symbol: str,  timeframe: str, upto_date: str,
                               expiry_date_label: str, option_type: str, root_dir: str = CURRENT_FILE_DIR):
    return Path(f"{root_dir}/data/{str(exchange).lower()}/{underlying_symbol}/upto{upto_date}"
                f"/{expiry_date_label}/{option_type}/{timeframe}")

def get_dir_spot_price_data_path(exchange, underlying_symbol,
                             upto_date, timeframe, root_dir=CURRENT_FILE_DIR):
       return Path(f"{root_dir}/data/{str(exchange).lower()}/{underlying_symbol}/upto{upto_date}/"
                f"underlying/{timeframe}")


def get_option_price_data_path(exchange, underlying_symbol, base_curr, nom_currency, timeframe, upto_date,
                               expiry_date_label, option_type, exp_price, root_dir=CURRENT_FILE_DIR):
    dir_path = get_dir_option_price_data_path(exchange, underlying_symbol, timeframe, upto_date, expiry_date_label,
                                              option_type, root_dir=root_dir)
    type_label = 'C' if option_type == 'call' else 'P'
    filename = f"{underlying_symbol}_{base_curr}_{nom_currency}-{expiry_date_label}-{exp_price}-{type_label}.csv"
    return dir_path/Path(filename)

def get_spot_price_data_path(exchange, underlying_symbol, base_curr, nom_currency,
                             upto_date, timeframe, root_dir=CURRENT_FILE_DIR):
    dir_path = get_dir_spot_price_data_path(exchange, underlying_symbol, upto_date, timeframe, root_dir=root_dir)
    filename = f"{underlying_symbol}_{base_curr}_{nom_currency}.csv"

    return dir_path/Path(filename)

def load_option_ohlcv_from_csv(exchange, underlying_symbol, timeframe, upto_date, expiry_date_label, option_type,
                               root_dir=CURRENT_FILE_DIR):
    file_path = get_option_price_data_path(exchange, underlying_symbol, timeframe, upto_date,
                                           expiry_date_label, option_type, root_dir=root_dir)
    return load_ohlcv_from_csv(file_path)

def load_underlying_ohlcv_from_csv(exchange, underlying_symbol, upto_date, timeframe, root_dir=CURRENT_FILE_DIR):
    file_path = get_spot_price_data_path(exchange, underlying_symbol, upto_date, timeframe, root_dir=root_dir)
    return load_ohlcv_from_csv(file_path)

def load_ohlcv_from_csv(file_path)-> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

if __name__ == "__main__":
    # scrape_candles_to_csv('BTC-USDT__USDT-250627-55000-C_1d.csv', 'binance',
    #                       'BTC/USDT:USDT-250627-55000-C', '1d',
    #                       '2025-01-0100:00:00Z', limit=1000)
    # option_contracts = load_all_option_metadata_for_symbol(get_exchange('deribit'), 'BTC')
    # pprint(option_contracts.keys())
    # print("Number of markets:", len(option_contracts))
    # greeks = get_exchange('binance').fetch_greeks('BTC/USDT:USDT-250627-55000-C')
    # pprint(greeks)
    #download_option_prices(get_exchange('binance'), 'BTC', base_curr='USDT', nom_currency='USDT',
    #                        timeframe='1d', expiry_date_label='250708', since_date='2023-01-01 00:00:00Z')
    download_option_and_spot_prices("deribit", 'BTC', '1d',
                                   '250708', since_date='2020-01-01 00:00:00Z', upto_date='250708')
    # download_spot_prices(get_exchange('deribit'), 'BTC', base_curr='USD', nom_currency='BTC',
    #                     timeframe='1d', since_date='2025-01-01')
    #xchange = get_exchange('binance')
    #ohlcv = load_ohlcv(exchange, 'BTC/USDT', '1d')
    # ticker = exchange.fetch_ticker('BTC/USDT')
    # pprint(ticker)


    # df = load_option_ohlcv_from_csv(get_exchange('binance'), 'BTC', '1d', '2024-01-01 00:00:00Z',
    #                                 '250627', 'call')
    # print(df)

    #exchange = get_exchange('deribit')
    #prices = exchange.fetch_ohlcv('BTC/USD:BTC', '1d')
    #print(prices)
    # all_markets = exchange.fetch_markets()  #params={'type': 'spot', 'limit': 100000}
    # print(all_markets)
