
import csv
import pandas as pd
import datetime
import numpy as np
from mains.combine_data import DataCombiner

data_dir = '../data'
djia_file = '{}/DJIA_2014.csv'.format(data_dir)
sp_file = '{}/SP_2014.csv'.format(data_dir)
news_file = '{}/uci-news-aggregator_small.csv'.format(data_dir)


def main():
    '''
    '''
    # embeddings = pd.read_csv('../data/embedding_results_small.csv')
    stock_dtype = {'Open': np.float32, 'Close': np.float32}
    djia = pd.read_csv(djia_file, dtype=stock_dtype)
    sp = pd.read_csv(sp_file, dtype=stock_dtype)

    news = pd.read_csv(news_file)
    res = news['TIMESTAMP'].apply(convert_time).to_frame().join(news['TITLE'])

    existing_dates = set(djia['Date'].unique()).intersection(set(sp['Date'].unique()))

    mask = res['TIMESTAMP'].isin(existing_dates)
    valid_entries = res[mask]

    # filter weekends (missing dates in djia/sp)

    djia_labels, djia_deltas = compute_label_delta(djia)
    sp_labels, sp_deltas = compute_label_delta(sp)

    output = pd.concat([djia_labels, djia_deltas, sp_labels, sp_deltas], axis=1)
    output.to_csv('../data/market_headline_embeddings_small.csv',
                  header=['djia_label', 'djia_delta', 'sp_label', 'sp_delta'],
                  index=False)


def compute_label_delta(stock):
    deltas = stock['Open'] - stock['Close']
    labels = deltas.copy().apply(lambda d: int(d > 0)).astype(int)
    return labels, deltas


def convert_time(timestamp):
    s = int(timestamp) / 1000
    return datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d')


def __findStockChange(self, row, dataset, dateOffset=0):
    date = datetime.strptime(row, '%Y-%m-%d')
    date = date + datetime.timedelta(days=dateOffset)
    row = date.strftime('%Y-%m-%d')
    currentstockDay = dataset[dataset['Date'] == row]
    if not currentstockDay.empty:
        return currentstockDay.iloc[0]['Close'] > currentstockDay.iloc[0]['Open']
    else:
        return False


if __name__ == '__main__':
    main()
