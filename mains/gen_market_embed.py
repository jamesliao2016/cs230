# Produces file called market_headline_embeddings_*.tsv
# labels, deltas, embeddings (512)

import csv
import pandas as pd
import datetime
import numpy as np

def main():
    djia = pd.read_csv('../data/DJIA_2014.csv', dtype={'Open': np.float32, 'Close': np.float32})
    sp = pd.read_csv('../data/SP_2014.csv', dtype={'Open': np.float32, 'Close': np.float32})
    djia_labels, djia_deltas = compute_label_delta(djia)
    sp_labels, sp_deltas = compute_label_delta(sp)
    embeddings = pd.read_csv('../data/embedding_results_small.csv')

    output = pd.concat([djia_labels, djia_deltas, sp_labels, sp_deltas, embeddings], axis=1)
    output.to_csv('../data/market_headline_embeddings_small.csv',
                  header=False,
                  index=False)


def compute_label_delta(stock):
    deltas = stock['Open'] - stock['Close']
    labels = deltas.copy().apply(lambda d: int(d > 0)).astype(int)
    return labels, deltas


if __name__ == '__main__':
    main()
