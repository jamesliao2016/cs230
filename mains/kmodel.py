from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard
from mains.project_data import get_headline_embeddings

import numpy as np
import pandas as pd

# fix random seed for reproducibility
np.random.seed(7)

n_embed = 512

data_dir = '../data'
dataset_file = '{}/combined_result.tsv'.format(data_dir)
embeddings_file = '{}/embedding_results.csv'.format(data_dir)

def main():
    # load csv with combined news headlines, deltas as labels
    # label, delta, embedding(512)

    dataset = pd.read_table(dataset_file)
    headlines = dataset['headline'].values
    labels = dataset['sp_label'].values

    embeddings = load_embeddings(headlines)

    # create model
    model = Sequential()
    model.add(Dense(128, input_shape=(None, n_embed), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # checkpoint
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    tb_callback = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    callbacks_list = [checkpoint, tb_callback]

    # Fit the model
    model.fit(embeddings, labels, validation_split=0.2, epochs=200, batch_size=32, callbacks=callbacks_list)

    # evaluate the model
    scores = model.evaluate(embeddings, labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def load_embeddings(headlines):
    print('Loading embeddings...')
    # return np.loadtxt(embeddings_file, dtype=np.float32, delimiter=', ')
    result = get_headline_embeddings(headlines)
    print('Finished loading embeddings')
    return result


if __name__ == '__main__':
    main()
