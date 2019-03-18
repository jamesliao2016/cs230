import gensim
import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K
import keras_metrics as km

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Embedding, LSTM, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard
from gensim.models.keyedvectors import KeyedVectors

DEBUG = False

day_offset = 2

data_dir = '../data'
dataset_file = '{}/combined_result_day_offset_{}{}.tsv'.format(data_dir, day_offset, '_small' if DEBUG else '')

model_name = 'w2v/day_offset_{}'.format(day_offset)
tb_log_dir = '../experiments/{}'.format(model_name)


def main():
    print('Loading dataset: {}'.format(dataset_file))
    dataset = pd.read_table(dataset_file)

    headlines = dataset['title'].values
    labels = dataset['sp_label'].values

    # checkpoint
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    tb_callback = TensorBoard(log_dir=tb_log_dir, histogram_freq=0, write_graph=True, write_images=True)

    model, y_pred = create_model(headlines, embed_type='word')

    # Fit the model
    model.fit(y_pred, labels, validation_split=0.2, epochs=200, batch_size=32, callbacks=[checkpoint, tb_callback])

    # evaluate the model
    scores = model.evaluate(y_pred, labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


class WordModel:
    def __init__(self, headlines):
        self.headlines = headlines


def create_model(headlines, embed_type='word'):
    def setup_word_model():
        sentences = [[w for w in h.lower().split()] for h in headlines]

        # TODO: google word2vec
        # word_vectors = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

        word_model = gensim.models.Word2Vec(sentences, size=100, min_count=1, window=5, iter=100)
        pretrained_weights = word_model.wv.syn0
        vocab_size, emdedding_size = pretrained_weights.shape

        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
        model.add(LSTM(units=emdedding_size))
        model.add(Dense(units=vocab_size, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', km.binary_precision(), km.binary_recall()])

        return model, pretrained_weights

    def setup_sentence_model():
        embeddings = get_sentence_embeddings(headlines)
        model = Sequential()
        model.add(Dense(256, input_dim=512, activation='relu', bias_initializer='zeros'))
        model.add(Dense(1, activation='sigmoid', bias_initializer='zeros'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', km.binary_precision(), km.binary_recall()])
        return model, embeddings

    return setup_word_model() if embed_type == 'word' else setup_sentence_model()


def penalized_loss(noise):
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) - K.square(y_true - noise), axis=-1)
    return loss


def get_sentence_embeddings(headlines):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

    # Import the Universal Sentence Encoder's TF Hub module
    print('Getting Hub Module')
    os.environ['TFHUB_CACHE_DIR'] = '/home/ubuntu/cs230-final-ralmodov/tf_cache'
    embed = hub.Module(module_url)
    print('Finished getting Hub Module')

    return run_embed(embed, headlines)


def run_embed(embed, headlines):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        return session.run(embed(headlines))


if __name__ == '__main__':
    main()
