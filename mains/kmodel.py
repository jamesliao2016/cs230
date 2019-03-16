import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import datetime as dt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, TensorBoard

data_dir = '../data'
dataset_file = '{}/combined_result.tsv'.format(data_dir)
embeddings_file = '{}/embedding_results.csv'.format(data_dir)

model_name = 'day_offset_2'
tb_log_dir = '../experiments/univ/{}'.format(model_name)


def main():
    dataset = pd.read_table(dataset_file)
    train_set, dev_set, test_set = split_train_dataset(dataset)

    # input
    embeddings = load_embeddings(train_set['title'].values)

    # output
    labels = get_labels(dataset, label='sp_label', day_offset=2)

    # create model
    model = Sequential()
    model.add(Dense(256, input_dim=512, activation='relu', bias_initializer='zeros'))
    model.add(Dense(1, activation='sigmoid', bias_initializer='zeros'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # checkpoint
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    tb_callback = TensorBoard(log_dir=tb_log_dir, histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [checkpoint, tb_callback]

    # Fit the model
    model.fit(embeddings, labels, epochs=200, batch_size=32, callbacks=callbacks_list)

    # evaluate the model
    scores = model.evaluate(dev_set, labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def get_labels(dataset, label, day_offset=0):
    labels = dataset[label].values

    def day_delta(day):
        return dt.strptime(day, '%Y-%m-%d') + dt.timedelta(days=day_offset)

    return labels['date'].apply(day_delta)


def load_embeddings(headlines):
    print('Loading embeddings...')
    # result = np.loadtxt(embeddings_file, dtype=np.float32, delimiter=', ')
    result = fetch_headline_embeddings(headlines)
    print('Finished loading embeddings')
    return result


def fetch_headline_embeddings(headlines):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"  # @param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

    # Import the Universal Sentence Encoder's TF Hub module
    print('Getting Hub Module')
    embed = hub.Module(module_url)
    print('Finished getting Hub Module')

    return run_embed(embed, headlines)


def run_embed(embed, headlines):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        return session.run(embed(headlines))


def split_train_dataset(dataset):
    shuffled = dataset.sample(frac=1, random_state=7)  # shuffles the ordering of filenames (deterministic given the chosen seed)

    n_d = len(shuffled)
    split_1 = int(0.8 * n_d)
    split_2 = int(0.9 * n_d)
    train = shuffled[:split_1]
    dev = shuffled[split_1:split_2]
    test = shuffled[split_2:]
    return train, dev, test


if __name__ == '__main__':
    main()
