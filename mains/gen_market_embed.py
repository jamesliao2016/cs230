# Produces file called market_headline_embeddings_*.tsv
# labels, deltas, embeddings (512)

import csv
import pandas
import datetime


def main():

    embeddings = pandas.read_csv('../data/embedding_results_small.csv')

    djia = pandas.read_csv('../data/DJIA_2014.csv')
    sp = pandas.read_csv('../data/SP_2014.csv')
    djia_deltas = compute_deltas(djia)


    # small file
    with open('../data/market_headline_embeddings_small.tsv', 'w') as f:
        f.write('{}\t{}\t{}'.format())


def compute_deltas(row, dataset, dateOffset=0):
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
