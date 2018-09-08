"""
Joseph Acevedo
15 June, 2018

Test of creating a 2D vector word map using Tensorflow's prebuilt
'word2vec' framework for the HNHLC project.
"""

from __future__ import absolute_import, division, print_function
import codecs
import io
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime

nltk.download("punkt")
nltk.download("stopwords")

book_filenames = sorted(glob.glob("/*.txt"))
print("[{0}]Found books: ".format(datetime.datetime.now().strftime("%H:%M.%S")))
book_filenames

# Step 1: process data

corpus_raw = u""
for book_filename in book_filenames:
    print("[{0}]Reading '{1}'...".format(datetime.datetime.now().strftime("%H:%M.%S"),book_filename))
    with io.open(book_filename, 'r', encoding='utf8') as book_file:
        corpus_raw += book_file.read()
    print("[{0}]Corpus is now {1} characters long".format(datetime.datetime.now().strftime("%H:%M.%S"),len(corpus_raw)))
    print()

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

raw_sentences = tokenizer.tokenize(corpus_raw)

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ",raw)
    words = clean.split()
    return words

sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

"""
print(raw_sentences[5])
print(sentence_to_wordlist(raw_sentences[5]))
"""

token_count = sum([len(sentence) for sentence in sentences])
print("[{0}]The book corpus contains {1:,} tokens".format(datetime.datetime.now().strftime("%H:%M.%S"),token_count))


# Dimensionality of the resulting word vectors.
#more dimensions mean more traiig them, but more generalized
num_features = 300

# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#how often to use
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
seed = 1

wap2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

wap2vec.build_vocab(sentences)
print("[{0}]Word2Vec vocabulary length: ".format(datetime.datetime.now().strftime("%H:%M.%S")), len(wap2vec.wv.vocab))

# Train model on sentences
wap2vec.train(sentences, total_examples=wap2vec.corpus_count,epochs=wap2vec.epochs)

print("[{}]Training Complete".format(datetime.datetime.now().strftime("%H:%M.%S")))

# Save model
if not os.path.exists("trained"):
    os.makedirs("trained")

wap2vec.save(os.path.join("trained", "wap2vec.w2v"))

# Load model
wap2vec = w2v.Word2Vec.load(os.path.join("trained", "wap2vec.w2v"))

# Reduce dimensionality for visualization
print("[{}]Reducing Dimensionality".format(datetime.datetime.now().strftime("%H:%M.%S")))
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
print("[{}]Dimensionality Reduced".format(datetime.datetime.now().strftime("%H:%M.%S")))

all_word_vectors_matrix = wap2vec.wv.syn0

print("[{}]Transform Fitting. May take about 5 minutes".format(datetime.datetime.now().strftime("%H:%M.%S")))
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
print("[{}]Transform Fitted".format(datetime.datetime.now().strftime("%H:%M.%S")))

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[wap2vec.wv.vocab[word].index])
            for word in wap2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)

points.head(10)

sns.set_context("poster")

points.plot.scatter("x", "y", s=10, figsize=(20, 12))

plt.show()

