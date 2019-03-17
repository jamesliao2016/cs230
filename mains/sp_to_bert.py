import pandas as pd

DEBUG = False

day_label_offset = 2

data_dir = '../data'
dataset_file = '{}/combined_result_day_offset_{}{}.tsv'.format(data_dir, day_label_offset, '_small' if DEBUG else '')
in_file = '{}/combined_result_day_offset_{}.tsv'.format(data_dir, day_label_offset)

out_dir = '../glue_data/SP'.format(day_label_offset)


def main():
    print('Loading dataset: {}'.format(dataset_file))
    dataset = pd.read_csv(dataset_file, sep='\t')

    train, dev, test = split_train_dataset(dataset)

    write_output(train, 'train.tsv')
    write_output(dev, 'dev.tsv')
    write_output(test, 'test.tsv', is_test=True)


def write_output(data, filename, is_test=False):
    out_file = '{}/{}'.format(out_dir, filename)
    if is_test:
        data['title'].rename('sentence').to_frame().to_csv(out_file, sep='\t', index=True)
    else:
        data['title'].rename('sentence').to_frame().join(data['sp_label'].rename('label').to_frame()).to_csv(out_file, sep='\t', index=False)
    print('Wrote contents to file: {}'.format(out_file))


def split_train_dataset(dataset):
    shuffled = dataset.sample(frac=1, random_state=7)  # shuffles the ordering of filenames (deterministic given the chosen seed)

    n_d = len(shuffled)
    split_1 = int(0.8 * n_d)
    split_2 = int(0.9 * n_d)
    train = shuffled[:split_1]
    dev = shuffled[split_1:split_2]
    test = shuffled[split_2:]
    return train, dev, test


if __name__ == "__main__":
    main()
