from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

import numpy as np

# fix random seed for reproducibility
np.random.seed(7)

n_embed = 512


def main():
    # load csv with combined news headlines, deltas as labels
    # label, delta_tmrw, embedding(512)
    dataset = np.loadtxt("data/market_headline_embeddings_small.tsv", delimiter='\t')

    # split into input (X) and output (Y) variables
    labels, deltas, embeddings = partition_data(dataset)

    # create model
    model = Sequential()
    model.add(Dense(128, input_shape=(None, n_embed), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    # TODO will require functional model when adding delta as loss
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # checkpoint
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Fit the model
    model.fit(embeddings, labels, epochs=150, batch_size=10, callbacks=callbacks_list)

    # evaluate the model
    scores = model.evaluate(embeddings, labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def partition_data(dataset):
    labels = dataset[:, 0]
    deltas = dataset[:, 1]
    embeddings = dataset[:, 2:]
    return labels, deltas, embeddings


if __name__ == '__main__':
    main()
