
import csv
import pandas as pd
import datetime
import numpy as np
from mains.combine_data import DataCombiner

data_dir = '../data'
djia_file = '{}/DJIA_2014.csv'.format(data_dir)
sp_file = '{}/SP_2014.csv'.format(data_dir)
news_file = '{}/uci-news-aggregator_small.csv'.format(data_dir)
out_file = '{}/output.tsv'.format(data_dir)


def main():
    # embeddings = pd.read_csv('../data/embedding_results_small.csv')
    stock_dtype = {'Open': np.float32, 'Close': np.float32}
    djia = pd.read_csv(djia_file, dtype=stock_dtype)
    sp = pd.read_csv(sp_file, dtype=stock_dtype)

    news = pd.read_csv(news_file)
    res = news['TITLE'].to_frame().join(news['TIMESTAMP'].apply(convert_time).to_frame())

    existing_dates = set(djia['Date'].unique()).intersection(set(sp['Date'].unique()))

    djia_sp = djia[['Date', 'Open', 'Close']]\
        .rename(index=str, columns={'Open': 'djia_open', 'Close': 'djia_close'})\
        .join(sp[['Open', 'Close']].rename(index=str, columns={'Open': 'sp_open', 'Close': 'sp_close'}))

    mask = res['TIMESTAMP'].isin(existing_dates)
    news_final = res[mask]

    stocks = concat_label_delta(djia_sp)

    print('done')
    # output = pd.concat(news_final, stocks[]['djia_label'])
    # output.to_csv(out_file,
    #               sep='\t',
    #               header=['headline', 'date', 'djia_label', 'djia_delta', 'sp_label', 'sp_delta'],
    #               index=False)


def concat_label_delta(stock):
    djia_delta = stock['djia_open'] - stock['djia_close']
    sp_delta = stock['sp_open'] - stock['sp_close']
    djia_label = djia_delta.copy().apply(lambda d: int(d > 0)).astype(int)
    sp_label = sp_delta.copy().apply(lambda d: int(d > 0)).astype(int)
    return stock.join([djia_label.rename('djia_label'), djia_delta.rename('djia_delta'), sp_label.rename('sp_label'), sp_delta.rename('sp_delta')])


def convert_time(timestamp):
    s = int(timestamp) / 1000
    return datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d')


def __findStockChange(row, dataset):
    currentstockDay = dataset[dataset['Date'] == row]
    if not currentstockDay.empty:
        return currentstockDay.iloc[0]['djia_close'] > currentstockDay.iloc[0]['djia_open']
    else:
        return False


if __name__ == '__main__':
    main()
