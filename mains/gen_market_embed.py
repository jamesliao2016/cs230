# Produces file called market_headline_embeddings_*.tsv
# labels, deltas, embeddings (512)

import csv

def main():

    with open('data/embedding_results.csv') as f:
        embeddings = list(csv.reader(f, delimiter=','))

    # small file
    with open('data/market_headline_embeddings_small.tsv') as f:


def get_

if __name__ == '__main__':
    main()
