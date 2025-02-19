import re
import string

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

import torch
import transformers

def preprocess(text):
	"""
	text: pd.Series

	returns doc[], where doc = token[]
	"""
	# lowercase text
	lowered = text.str.lower() # use Dataset B

	# remove punctuation
	punct_re = re.compile("[{}]".format(re.escape(string.punctuation)))
	punct_free = lowered.str.replace(punct_re, "", regex=True)
	punct_free

	# tokenize by sentence first (sent_tokenize can help differentiate sentence better than word_tokenize, e.g. "Mr. Smith")
	sent_tokens = punct_free.apply(sent_tokenize)

	# split each sentence into individual word tokens
	word_tokens = []
	for doc in sent_tokens.values:
	    tokens = []
	    for sent in doc:
	        tokens.extend(word_tokenize(sent)) # extend to preserve 1D structure
	    word_tokens.append(tokens)
	word_tokens = pd.Series(word_tokens, index=sent_tokens.index)

	# perform POS tagging (before stopwords removal)
	pos_tags = word_tokens.apply(nltk.pos_tag) # perform POS tagging before stopwords removal to allow more context for tagger to tag correctly

	# remove tokens that are stopwords, excluding words that negate the meaning
	stopwords_set = set(stopwords.words("english"))
	negate_words_set = set(("no", "nor", "not", "isn't", "wasn't", "mustn't", "don", "couldn't", "shouldn", "mustn", "isn", "doesn't", "hadn't", "isn't", "mightn", "doesn", "needn't", "wouldn't", "haven", "shouldn't", "hadn", "needn", "aren't", "weren", "mightn't" "haven't", "didn", "weren't", "hasn't", "shan", "aren", "didn't", "shan't"))
	final_set = stopwords_set.difference(negate_words_set)
	stopwords_free = pos_tags.apply(lambda doc: [token for token in doc if token[0] not in final_set])

	# strip non-alphanumeric characters
	nonalphanum_re = re.compile("[^a-zA-Z]") # only allow alphabetical characters (also strip numerical characters since they provide no contextual meaning)
	nonalphanum_free = stopwords_free.apply(lambda doc: [(nonalphanum_re.sub("", token[0]).strip(), token[1]) for token in doc])

	# remove tokens with only 2 characters or less
	lt_2_free = nonalphanum_free.apply(lambda doc: [token for token in doc if len(token[0]) >= 3])

	# last step: lemmatise tokens with POS tag
	# POS tag arg of .lemmatize() only accepts "n", "v", "a", "r", "s" # https://www.nltk.org/api/nltk.stem.WordNetLemmatizer.html?highlight=wordnet
	lemmatise_pos_tag_arg_map = {"n": "n", "v": "v", "j": "a", "r": "r", "s": "s"}
	lemmatise_pos_tag_arg_mapper = lambda x: lemmatise_pos_tag_arg_map[x] if x in lemmatise_pos_tag_arg_map else "n" 
	lemmatiser = WordNetLemmatizer()

	lemmatised_tokens = lt_2_free.apply(lambda doc: [(lemmatiser.lemmatize(token[0], lemmatise_pos_tag_arg_mapper(token[1][0].lower())), token[1]) for token in doc])

	# obtain chunking dictionary from initial dataset
	common_bigrams = np.load("data/common_bigrams.npy")

	# drop POS tags here
	tokens = lemmatised_tokens.apply(lambda doc: [token[0] for token in doc]) # drop POS tags

	# perform chunking
	mwe_tokeniser = MWETokenizer(common_bigrams.tolist())
	chunked_tokens = tokens.apply(lambda doc: mwe_tokeniser.tokenize(doc))

	return chunked_tokens


class DistilBERTCLSEmbeddingTransformer(BaseEstimator, TransformerMixin):
	def __init__(self, verbose=0):
		self.model_name = "distilbert-base-uncased"
		self.tokeniser = transformers.DistilBertTokenizer.from_pretrained(self.model_name) # load tokenizer from pretrained weights
		self.model = transformers.DistilBertModel.from_pretrained(self.model_name) # load DistilBERT model from pretrained weights

		# process data in batches else out of memory
		self.batch_size = 512

		# misc
		self.verbose = verbose

	def fit(self, X, y=None):
		return self # no fitting required as no learning process, using pre-trained weights

	def transform(self, X, y=None):
		"""
		X: pd.Series, test sample, assumes documents are NOT pre-processed
		y: None, not needed, but will be supplied as part of pipeline

		returns float[][], list of document vectors with each document vector having size self.vector_size
		"""
		embeddings = []
		for i in range(0, len(X), self.batch_size):
			doc_batch = X[i: min(i +self.batch_size, len(X))]
			if self.verbose >= 1: print("Batch_idx={}".format(i))

			# add_special_tokens to include [CLS] and [SEP] at first and last position of sentence
			inputs = self.tokeniser( # batch encode (tokenise + padding, stored in ["input_ids"] and ["attention_mask"] attribute of return value)
				doc_batch, # tokenise batch
				add_special_tokens=True,
				max_length=512,
				truncation=True, # truncate to include 512 tokens only
				padding=True, # automatically pad tokens to max length encountered
				return_tensors="pt" # pytorch tensors
			)

			# obtain embeddings from pretokenizer
			with torch.no_grad():
				outputs = self.model(
					input_ids=inputs["input_ids"],
					attention_mask=inputs["attention_mask"]
				)
		
				cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy() # extract [CLS] token embeddings
				embeddings.extend(cls_embeddings.tolist())
		return embeddings

class FinalEstimator:
	random_state = 100

	def __init__(self):
		# read X and y
		data = pd.read_csv("data/Group1_2_DatasetA.csv")
		y = data["Sentiment"] # expects series

		# remove neutral class
		y = y.loc[~(y == "neutral")]
		print(y.shape)

		# read training data embeddings
		embeddings_np = [] # store embeddings
		for i in range(0, y.shape[0], 512):
			embeddings_np.append(np.load("data/distilbert/embeddings_{}.npy".format(i)))
		X_raw_embeddings = np.concatenate(embeddings_np)

		# fit logistic regression
		approach_4 = Pipeline([
			("oversample", SMOTE(random_state=self.__class__.random_state)),
			("scaler", StandardScaler()),
			("classifier", LogisticRegression(C=0.05, max_iter=1000)) # obtained C from hyper-parameters tuning
		])
		self.fitted = approach_4.fit(X_raw_embeddings, y)

		# define preprocessing layer
		self.embedding = DistilBERTCLSEmbeddingTransformer()

	def predict(self, text):
		"""
		text: str

		return sentiment: str, tokens: str[], probability_neg: float, probability_pos: float
		"""
		embedding = self.embedding.transform([text]) # wrap in 2D structure to run inference
		predicted_label = self.fitted.predict(embedding)[0]
		tokens = preprocess(pd.Series([text]))
		predicted_proba = self.fitted.predict_proba(embedding)[0]
		return predicted_label, tokens[0], predicted_proba[0], predicted_proba[1] # return for first document
