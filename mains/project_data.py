
# # Process data from sources


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv


# ## Convert headlines to embeddings

news_file = '../data/uci-news-aggregator.csv'
# news_file = '../data/uci-news-aggregator_small.csv'


def main():
    news = read_news()
    headlines = [n[1] for n in news[1:]]

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

    # Import the Universal Sentence Encoder's TF Hub module
    print('Getting Hub Module')
    embed = hub.Module(module_url)
    print('Finished getting Hub Module')

    run_embed(embed, headlines)


def run_embed(embed, headlines):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        headline_embeddings = session.run(embed(headlines))

        for i, hemb in enumerate(np.array(headline_embeddings).tolist()):
            print("Message: {}".format(headlines[i]))
            print("Embedding size: {}".format(len(hemb)))
            hemb_snippet = ", ".join(
                (str(x) for x in hemb[:3]))
            print("Embedding: [{}, ...]\n".format(hemb_snippet))


def iter_csv(file_name, task_indexed):
    with open(file_name) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if idx != 0:
                task_indexed(idx, row)


def read_news():
    with open(news_file) as csv_file:
        return list(csv.reader(csv_file, delimiter=','))


if __name__ == "__main__":
    main()

