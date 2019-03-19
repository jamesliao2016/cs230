import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras_metrics as km
import argparse

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, Embedding, LSTM, Activation, Reshape, Conv2D, \
    MaxPooling2D, concatenate, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras import regularizers, Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

parser = argparse.ArgumentParser()
parser.add_argument('--embed_type', default='word',
                    help="'word' or 'sentence' for embedding type")
parser.add_argument('--model_name', default='w2v',
                    help="Name of the model")
parser.add_argument('--day_offset', default=0,
                    help="Offset the labels by given day num")
parser.add_argument('--data_dir', default='../data',
                    help="Directory where to find data file")
parser.add_argument('--glue_data_dir', default='../glue_data',
                    help="Directory where to find train dev test files")
parser.add_argument('--batch_size', default=64,
                    help="Batch size")
parser.add_argument('--lambd', default=1e-3,
                    help="L2 regularization term")

args = parser.parse_args()

pretrained_w2v_file = '{}/GoogleNews-vectors-negative300.bin'.format(args.data_dir)
train_file = '{}/SP_day_offset_{}/train.tsv'.format(args.glue_data_dir, args.day_offset)
eval_file = '{}/SP_day_offset_{}/dev.tsv'.format(args.glue_data_dir, args.day_offset)
tb_log_dir = '../experiments/{}'.format(args.model_name)


def main():
    train_set = pd.read_table(train_file)
    eval_set = pd.read_table(eval_file)

    headlines_train = train_set['sentence'].values
    labels_train = train_set['label'].values
    assert headlines_train.shape == labels_train.shape

    headlines_eval = eval_set['sentence'].values
    labels_eval = eval_set['label'].values
    assert headlines_eval.shape == labels_eval.shape

    checkpoint = ModelCheckpoint(filepath="weights.best.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    tb_callback = TensorBoard(log_dir=tb_log_dir, histogram_freq=0, write_graph=True, write_images=True)

    model = create_model(headlines_train, headlines_eval, labels_train, labels_eval, embed_type=args.embed_type, callbacks=[checkpoint, tb_callback])


def create_model(headlines_train, headlines_eval, labels_train, labels_eval, embed_type, callbacks):
    def setup_word_model():
        word_vectors = KeyedVectors.load_word2vec_format(pretrained_w2v_file, binary=True)

        NUM_WORDS = 20000
        tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                              lower=True)
        tokenizer.fit_on_texts(headlines_train)
        sequences_train = tokenizer.texts_to_sequences(headlines_train)
        sequences_valid = tokenizer.texts_to_sequences(headlines_eval.text)
        word_index = tokenizer.word_index

        X_train = pad_sequences(sequences_train)
        X_eval = pad_sequences(sequences_valid, maxlen=X_train.shape[1])

        EMBEDDING_DIM = 300
        embedding_matrix = np.zeros((NUM_WORDS, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= NUM_WORDS:
                break
            try:
                embedding_matrix[i] = word_vectors[word]
            except KeyError:
                embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)

        del word_vectors

        sequence_length = X_train.shape[1]
        filter_sizes = [3, 4, 5]
        num_filters = 100
        drop = 0.5
        inputs = Input(shape=(sequence_length,))
        embedding = Embedding(NUM_WORDS, EMBEDDING_DIM, weights=[embedding_matrix], trainable=True)(inputs)
        reshape = Reshape((sequence_length, EMBEDDING_DIM, 1))(embedding)

        conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM), activation='relu',
                        kernel_regularizer=regularizers.l2(0.01))(reshape)
        conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM), activation='relu',
                        kernel_regularizer=regularizers.l2(0.01))(reshape)
        conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM), activation='relu',
                        kernel_regularizer=regularizers.l2(0.01))(reshape)

        maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1))(conv_0)
        maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1))(conv_1)
        maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1))(conv_2)

        merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
        flatten = Flatten()(merged_tensor)
        reshape = Reshape((3 * num_filters,))(flatten)
        dropout = Dropout(drop)(reshape)
        output = Dense(1, activation='sigmoid', bias_initializer='zeros', kernel_regularizer=regularizers.l2(args.lambd))(dropout)

        # this creates a model that includes
        model = Model(inputs, output)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', km.binary_precision(), km.binary_recall()])

        # Fit the model
        model.fit(X_train, labels_train, epochs=200, batch_size=args.batch_size, callbacks=callbacks, validation_data=(X_eval, labels_eval))

        return model

    def setup_sentence_model():
        X_train = get_sentence_embeddings(headlines_train)
        X_eval = get_sentence_embeddings(headlines_eval)
        model = Sequential()
        model.add(Dense(256, input_dim=512, activation='relu', bias_initializer='zeros', kernel_regularizer=regularizers.l2(args.lambd)))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid', bias_initializer='zeros'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', km.binary_precision(), km.binary_recall()])
        # Fit the model
        model.fit(X_train, labels_train, epochs=200, batch_size=args.batch_size, callbacks=callbacks, validation_data=(X_eval, labels_eval))
        return model

    return setup_word_model() if embed_type == 'word' else setup_sentence_model()


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
