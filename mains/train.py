import gensim
import os
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras_metrics as km
import argparse

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Embedding, LSTM, Activation
from keras.callbacks import ModelCheckpoint, TensorBoard
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras import regularizers
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
                    help="Directory where to find the data file")
parser.add_argument('--batch_size', default=32,
                    help="Batch size")
parser.add_argument('--lambd', default=1e-3,
                    help="L2 regularization term")

args = parser.parse_args()

pretrained_w2v_file = '{}/GoogleNews-vectors-negative300.bin'.format(args.data_dir)
train_file = '{}/combined_result_day_offset_{}.tsv'.format(args.data_dir, args.day_offset)
tb_log_dir = '../experiments/{}'.format(args.model_name)


def main():
    print('Loading dataset: {}'.format(dataset_file))
    dataset = pd.read_table(dataset_file)

    headlines = dataset['title'].values
    labels = dataset['sp_label'].values

    checkpoint = ModelCheckpoint(filepath="weights.best.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    tb_callback = TensorBoard(log_dir=tb_log_dir, histogram_freq=0, write_graph=True, write_images=True)

    model = create_model(headlines, embed_type=args.embed_type)

    # Fit the model
    model.fit(X_train, y_train, epochs=200, batch_size=args.batch_size, callbacks=[checkpoint, tb_callback])

    # evaluate the model
    scores = model.evaluate(X_eval, y_eval)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def create_model(headlines, embed_type='word'):
    def setup_word_model():
        word_vectors = KeyedVectors.load_word2vec_format(pretrained_w2v_file, binary=True)

        NUM_WORDS = 20000
        tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                              lower=True)
        tokenizer.fit_on_texts(texts)
        sequences_train = tokenizer.texts_to_sequences(texts)
        sequences_valid = tokenizer.texts_to_sequences(val_data.text)
        word_index = tokenizer.word_index

        EMBEDDING_DIM = 300
        vocabulary_size = 20000
        embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= vocabulary_size:
                break
            try:
                embedding_vector = word_vectors[word]
                embedding_matrix[i] = embedding_vector
            except KeyError:
                embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)

        del (word_vectors)

        from keras.layers import Embedding
        embedding_layer = Embedding(vocabulary_size,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    trainable=True)
        sequence_length = X_train.shape[1]
        filter_sizes = [3, 4, 5]
        num_filters = 100
        drop = 0.5
        inputs = Input(shape=(sequence_length,))
        embedding = embedding_layer(inputs)
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
        dropout = Dropout(drop)(flatten)
        output = Dense(units=3, activation='softmax', kernel_regularizer=regularizers.l2(args.lambd))(dropout)

        # this creates a model that includes
        model = Model(inputs, output)

        return model

    def setup_sentence_model():
        embeddings = get_sentence_embeddings(headlines)
        model = Sequential()
        model.add(Dense(256, input_dim=512, activation='relu', bias_initializer='zeros', kernel_regularizer=regularizers.l2(args.lambd)))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid', bias_initializer='zeros'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', km.binary_precision(), km.binary_recall()])
        return model, embeddings

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
