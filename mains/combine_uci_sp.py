# Combine UCI news aggregator dataset with S&P 500 and DJIA close

# Format:
# Date,Title,Hostname,Category,DJIA_Close,SP_Close,Delta_prev,Delta_next

import csv
import datetime

data_dir = '../data'
news_file = f'{data_dir}/uci-news-aggregator.csv'

whitelist = set('abcdefghijklmnopqrstuvwxyz 0123456789.,;\'-:?')


def main():
    djia, sp = read_djia_sp()
    news = read_news(djia, sp)
    write_output(news)


def read_djia_sp(djia_file=f'{data_dir}/DJIA_table.csv', sp_file=f'{data_dir}/SP_table.csv'):
    # Date to close
    djia = {}
    sp = {}

    iter_csv(djia_file, lambda idx, row: process_djia(djia, row))
    iter_csv(sp_file, lambda idx, row: process_sp(sp, row))

    return djia, sp


def iter_csv(file_name, task_indexed):
    with open(file_name) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if idx != 0:
                task_indexed(idx, row)


def process_djia(djia, row):
    date, _open, high, low, close, volume, adj_close = row
    djia[date] = close


def process_sp(sp, row):
    date, _open, high, low, close, adj_close, volume = row
    sp[date] = close


def read_news(djia, sp, header="date\ttitle\thostname\tcategory\tdjia_close\tsp_close"):
    rows = [header]
    with open(news_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for idx, row in enumerate(csv_reader):

            # REMOVE
            if idx > 10000:
                break

            if idx == 0:
                print(f'Column names are {", ".join(row)}')
            else:
                cleaned = process_news(djia, sp, row)
                rows.append(cleaned)
    return rows


def write_output(rows, out_file=f'{data_dir}/result.tsv'):
    t = '\t'
    with open(out_file, 'w') as f:
        for idx, r in enumerate(rows):
            f.write(f"{t.join(r) if idx != 0 else r}\n")


def process_news(djia, sp, row):
    """Return format: date,title,hostname,category,DJIA_Close,SP_Close"""

    id, title, url, publisher, category, story, hostname, timestamp = row
    date = convert_time(timestamp)
    return date, normalize_headline(title), hostname, category, djia[date], sp[date]


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
