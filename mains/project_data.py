#!/usr/bin/env python
# coding: utf-8

# # Process data from sources

# In[9]:


import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import csv


# ## Convert headlines to embeddings

# In[6]:


news_file = 'data/uci-news-aggregator.csv'


# In[14]:


def iter_csv(file_name, task_indexed):
    with open(file_name) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if idx != 0:
                task_indexed(idx, row)

def read_news():
    with open(news_file) as csv_file:
        return list(csv.reader(csv_file, delimiter=','))


# In[17]:


news = read_news()


# In[22]:


headlines = [n[1] for n in news[1:]][:10]

headlines[:10]  # check


# In[20]:


module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]


# In[24]:


# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  headline_embeddings = session.run(embed(headlines))

  for i, hemb in enumerate(np.array(headline_embeddings).tolist()):
    print("Message: {}".format(headlines[i]))
    print("Embedding size: {}".format(len(hemb)))
    hemb_snippet = ", ".join(
        (str(x) for x in hemb[:3]))
    print("Embedding: [{}, ...]\n".format(hemb_snippet))


# In[ ]:




