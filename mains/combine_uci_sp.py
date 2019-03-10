# Combine UCI news aggregator dataset with S&P 500 and DJIA close

# Format:
# Date,Title,Hostname,Category,SP_close,DJIA_close

import csv
import datetime

data_dir = '../data'


def main():
    with open(f'{data_dir}/uci-news-aggregator.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        line_count = 0
        for row in csv_reader:
            if line_count > 10:
                break

            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                cleaned = process_news(row)
                print(f'\t{cleaned}')
                line_count += 1

    print(f'Processed {line_count} lines.')


def process_news(row):
    """Return format: date,title,hostname,category"""

    id, title, url, publisher, category, story, hostname, timestamp = row
    return convert_time(timestamp), title, hostname, category


def convert_time(timestamp):
    s = timestamp / 1000.0
    return datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d')


if __name__ == "__main__":
    main()
