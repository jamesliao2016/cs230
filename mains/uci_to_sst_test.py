# Convert UCI news titles into test.tsv for BERT SST2

# Format:
# Date,Title,Hostname,Category,SP_close,DJIA_close

import csv
import datetime

data_dir = '../data'
whitelist = set('abcdefghijklmnopqrstuvwxyz 0123456789.,;\'-:?')


def main():
    with open(f'{data_dir}/uci-news-aggregator.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        rows = ['index\tsentence']  # header

        for idx, row in enumerate(csv_reader):

            if idx == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                out_row = f'{idx-1}\t{process_news(row)}'
                rows.append(out_row)

    write_output(rows)
    print('Done')


def write_output(rows):
    with open(f'{data_dir}/test.tsv', 'w') as f:
        for r in rows:
            f.write("%s\n" % r)


def process_news(row):
    """Return format: index,sentence"""

    id, title, url, publisher, category, story, hostname, timestamp = row
    return normalize_headline(title)


def convert_time(timestamp):
    s = timestamp / 1000
    return datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d')


def normalize_headline(row):
    result = row.lower()
    # Delete useless character strings
    result = result.replace('...', '')
    result = ''.join(filter(whitelist.__contains__, result))
    return result


if __name__ == "__main__":
    main()
