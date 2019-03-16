import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, TensorBoard

DEBUG = False

day_offset = 2

data_dir = '../data'
dataset_file = '{}/combined_result_day_offset_{}{}.tsv'.format(data_dir, day_offset, '_small' if DEBUG else '')
embeddings_file = '{}/embedding_results{}.csv'.format(data_dir, '_small' if DEBUG else '')

model_name = 'day_offset_{}'.format(day_offset)
tb_log_dir = '../experiments/univ/{}'.format(model_name)

os.environ['TFHUB_CACHE_DIR'] = '/home/ubuntu/cs230-final-ralmodov/tf_cache'


def main():
    print('Loading dataset: {}'.format(dataset_file))
    dataset = pd.read_table(dataset_file)

    headlines = dataset['title'].values
    labels = dataset['sp_label'].values
    embeddings = load_embeddings(headlines)

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
    model.fit(embeddings, labels, validation_split=0.2, epochs=200, batch_size=32, callbacks=callbacks_list)

    # evaluate the model
    scores = model.evaluate(embeddings, labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def penalized_loss(noise):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) - K.square(y_true - noise), axis=-1)
    return loss


def load_embeddings(headlines):
    print('Loading embeddings...')
    # result = np.loadtxt(embeddings_file, dtype=np.float32, delimiter=', ')
    result = fetch_headline_embeddings(headlines)
    print('Finished loading embeddings')
    return result


def fetch_headline_embeddings(headlines):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

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
