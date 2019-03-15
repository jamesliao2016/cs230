
# # Process data from sources


import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv



# ## Convert headlines to embeddings

data_file = '../data/combined_result.tsv'
small_data_file = '../data/combined_result_small.tsv'
out_file = '../data/embedding_results.csv'


def main():
    os.environ['TFHUB_CACHE_DIR'] = '/home/ubuntu/cs230-final-ralmodov/tf_cache'

    news = read_news()
    headlines = [n[1] for n in news[1:]]

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

    # Import the Universal Sentence Encoder's TF Hub module
    print('Getting Hub Module')
    embed = hub.Module(module_url)
    print('Finished getting Hub Module')

    headline_embeddings = run_embed(embed, headlines)
    write_output(headline_embeddings)


def write_output(headline_embeddings):
    with open(out_file, 'w') as f:
        for i, hemb in enumerate(np.array(headline_embeddings).tolist()):
            hemb_snippet = ", ".join((str(x) for x in hemb))
            f.write("{}\n".format(hemb_snippet))


def run_embed(embed, headlines):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        return session.run(embed(headlines))


def iter_csv(file_name, task_indexed):
    with open(file_name) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if idx != 0:
                task_indexed(idx, row)


def read_news():
    with open(data_file) as csv_file:
        return list(csv.reader(csv_file, delimiter=','))


if __name__ == "__main__":
    main()

