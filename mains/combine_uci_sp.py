# Combine UCI news aggregator dataset with S&P 500 and DJIA close

# Format:
# Date,Title,Hostname,Category,DJIA_Close,SP_Close,Delta_prev,Delta_next

import csv
import datetime

data_dir = '../data'
news_file = '{}/uci-news-aggregator.csv'.format(data_dir)
djia_file = '{}/DJIA_2014.csv'.format(data_dir)
sp_file = '{}/SP_2014.csv'.format(data_dir)
out_header = "date\ttitle\thostname\tcategory\tdjia_label\tdjia_delta\tsp_label\tsp_delta"
out_file = '{}/combined_result.tsv'.format(data_dir)


def main():
    djia, sp = read_djia_sp()
    news = read_news(djia, sp)
    write_output(news)


def read_djia_sp():
    # Date to close
    djia = {}
    sp = {}

    iter_csv(djia_file, lambda row: process_stock(djia, row))
    iter_csv(sp_file, lambda row: process_stock(sp, row))

    return djia, sp


def iter_csv(file_name, task_indexed):
    with open(file_name) as f:
        next(f)
        csv_reader = csv.reader(f)
        for row in csv_reader:
            task_indexed(row)


def process_stock(stock, row):
    date, _open, high, low, close, adj_close, volume = row
    delta = float(_open) - float(close)
    label = int(delta > 0)
    stock[date] = {
        'delta': str('{0:.6f}'.format(delta)),
        'label': str(label)
    }


def get_existing_dates(djia, sp):
    return set(djia.keys()).intersection(set(sp.keys()))


def read_news(djia, sp):
    news_rows = [out_header]
    existing_dates = get_existing_dates(djia, sp)
    with open(news_file) as f:
        next(f)
        csv_reader = csv.reader(f)
        for idx, row in enumerate(csv_reader):
            processed = process_news(existing_dates, djia, sp, row)
            if processed is not None:
                news_rows.append(processed)

    return news_rows


def write_output(rows):
    with open(out_file, 'w') as f:
        for idx, r in enumerate(rows):
            f.write('{}\n'.format('\t'.join(r) if idx != 0 else r))


def process_news(existing_dates, djia, sp, row):
    """
    Can return None if not in existing_dates
    """
    id, title, url, publisher, category, story, hostname, timestamp = row
    date = convert_time(timestamp)

    if date in existing_dates:
        return date, normalize_headline(title), hostname, category, djia[date]['label'], djia[date]['delta'], sp[date][
        'label'], sp[date]['delta']
    else:
        return None


def convert_time(timestamp):
    s = int(timestamp) / 1000
    return datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d')


def normalize_headline(row):
    # Delete useless character strings
    return row.replace('"', '').replace('\t', '')


if __name__ == "__main__":
    main()
