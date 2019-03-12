# Combine UCI news aggregator dataset with S&P 500 and DJIA close

# Format:
# Date,Title,Hostname,Category,SP_Close,DJIA_Close,Delta_prev,Delta_next

import csv
import datetime

data_dir = '../data'
news_file = f'{data_dir}/uci-news-aggregator.csv'
sp_file = f'{data_dir}/SP_table.csv'
djia_file = f'{data_dir}/DJIA_table.csv'

whitelist = set('abcdefghijklmnopqrstuvwxyz 0123456789.,;\'-:?')

def main():
    with open(news_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        rows = ["date\ttitle\thostname\tcategory"]

        for idx, row in enumerate(csv_reader):
            if idx > 100:
                break

            if idx == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                cleaned = process_news(row)
                rows.append(cleaned)

    write_output(rows)


def write_output(rows, out_file=f'{data_dir}/result.tsv'):
    with open(out_file, 'w') as f:
        for r in enumerate(rows):
            f.write("%s\n" % '\t'.join(r))


def process_news(row):
    """Return format: date,title,hostname,category"""

    id, title, url, publisher, category, story, hostname, timestamp = row
    return convert_time(timestamp), normalize_headline(title), hostname, category


def convert_time(timestamp):
    s = int(timestamp) / 1000
    return datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d')


def normalize_headline(row):
    result = row.lower()
    # Delete useless character strings
    result = result.replace('...', '')
    result = ''.join(filter(whitelist.__contains__, result))
    return result


if __name__ == "__main__":
    main()
