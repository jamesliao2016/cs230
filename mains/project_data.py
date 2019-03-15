
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

os.environ['TFHUB_CACHE_DIR'] = '/home/ubuntu/cs230-final-ralmodov/tf_cache'


def main():
    news = read_news()
    headlines = [n[1] for n in news[1:]]
    get_headline_embeddings(headlines)


def get_headline_embeddings(headlines):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

    # Import the Universal Sentence Encoder's TF Hub module
    print('Getting Hub Module')
    embed = hub.Module(module_url)
    print('Finished getting Hub Module')

    return run_embed(embed, headlines)
    # write_output(headline_embeddings)


def write_output(headline_embeddings):
    with open(out_file, 'w') as f:
        for i, hemb in enumerate(np.array(headline_embeddings).tolist()):
            hemb_snippet = ", ".join((str(x) for x in hemb))
            f.write("{}\n".format(hemb_snippet))


def run_embed(embed, headlines):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        return session.run(embed(headlines))


def read_news():
    with open(data_file) as f:
        return list(csv.reader(f, delimiter='\t'))


if __name__ == "__main__":
    main()

