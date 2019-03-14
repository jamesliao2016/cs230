# Produces file called market_headline_embeddings_*.tsv
# labels, deltas, embeddings (512)

import csv
import pandas as pd
import datetime
import numpy as np


def main():
    embeddings = pd.read_csv('../data/embedding_results_small.csv')

    djia = pd.read_csv('../data/DJIA_2014.csv', dtype={'Open': np.float32, 'Close': np.float32})
    sp = pd.read_csv('../data/SP_2014.csv', dtype={'Open': np.float32, 'Close': np.float32})
    djia_labels, djia_deltas = compute_label_delta(djia)
    sp_labels, sp_deltas = compute_label_delta(sp)


def compute_label_delta(stock):
    deltas = stock['Open'] - stock['Close']
    labels = deltas.apply(lambda d: int(d > 0))
    return labels, deltas


if __name__ == '__main__':
    main()
