import numpy as np
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import svm
from sklearn.model_selection import cross_val_score

data_dir = "./../../data/" 
data_file = "Combined_News_DJIA.csv"

def main():

	# Data
	print("Importing data...")
	data = import_data(data_dir + data_file)
	pre_processed = pre_process(data)

	# Model
	sentiment_included = analyze_sentiment(pre_processed)

	train_set, test_set = split_dataset(sentiment_included)

	train_labels = ravel_labels(train_set)
	test_labels = ravel_labels(test_set)

	train_sentiments = process_sentiments(train_set)
	test_sentiments = process_sentiments(test_set)

	print("Begin Training...")
	clsfr = train(train_sentiments, train_labels.astype('int'))

	results = cross_validate(test_sentiments, test_labels.astype('int'), clsfr)

	print("Test Label Mean: " + str(test_labels.mean()))
	print("Results: " + str(results))
	print("Avg sentiment: " + str(results.mean()))

def import_data(filename):
	return pd.read_csv(filename, header=0).fillna('').values

def pre_process(data):
	print("Preprocessing...")
	# Remove the leading 'b"' in front of all lines
	for row in data[0:475]:
		for field in row[2:]:
			if field:
				field = field[1:]
	return data

def analyze_sentiment(data):
	print("Analyzing sentiment...")
	sid = SentimentIntensityAnalyzer()
	avgs = np.empty((len(data),1))
	for i in range(0,len(data)):
		sentiments = []
		for field in data[i][2:]:
			sentiments.append(sid.polarity_scores(field)['compound'])
		avg = float(sum(sentiments))/len(sentiments)
		avgs[i] = avg
	return np.append(data, avgs, axis=1)

def split_dataset(data):
	# Split 80/10/10
	n_d = len(data)
	p_train = int(n_d * 0.8)
	p_dev = p_train + int(n_d * 0.1)

	train = data[:p_train]
	dev = data[p_train:p_dev]
	test = data[p_dev:]

	print("Train: {0} / Dev: {1} / Test: {2}".format(len(train), len(dev), len(test)))

	return train, test

def train(data, labels):
	return svm.SVC(kernel='linear', C=1).fit(data, labels)

def cross_validate(data, labels, clsfr):
	return cross_val_score(clsfr, data, labels, cv=5)

def ravel_labels(data):
	return data[:,1].ravel()

def process_sentiments(data):
	return data[:,27].reshape(len(data), 1)

if __name__ == "__main__":
	main()