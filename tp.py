# %%
import collections
from string import punctuation
from nltk import ngrams
from nltk import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
import json
import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from collections import Counter
import pandas as pd
# types of params

typeofclass = 1
lang = "en"
n = 2
c_path = "./corpora"
c_path = os.path.abspath(c_path)
# typeofclass = int(
#     input("to use word based classification use 1, for character based one use 2: "))
# lang = input("enter en for english corpus, ar for the arabic one: ")
# n = int(input("Enter the number of ngrams: "))
if lang == "en":
    c_path += "/english"
else:
    c_path += "/arabic"

stopwords = stopwords.words("english")

stoplist = set(stopwords + list(punctuation))

porter = PorterStemmer()


def preprocess_text(text: str):
    tokens = [token for token in nltk.word_tokenize(
        text) if token.lower() not in stoplist and not token.lower().isdigit()]
    tokens = [porter.stem(w) for w in tokens]
    return tokens


corpus_ngrams = []
corpus = PlaintextCorpusReader(c_path, ".*")
filesids = corpus._fileids
dictNgrams = {}
for f in filesids:
    file_sentences = corpus.sents(f)
    corpus_ngrams_perfile = []
    for sent in file_sentences:
        sent_words = preprocess_text(" ".join(sent))
        sent_n_grams = ngrams(sent_words, n)
        corpus_ngrams.append(list(sent_n_grams))
        corpus_ngrams_perfile += [w for w in corpus_ngrams[-1]]

    frq_dist_f = nltk.FreqDist(corpus_ngrams_perfile)
    temp_dict = {" ".join(k): v for k, v in dict(frq_dist_f).items()}
    temp_dict = collections.OrderedDict(sorted(temp_dict.items()))
    dictNgrams[f] = temp_dict

corpus_tokens = preprocess_text(corpus.raw())
tiidf = pd.DataFrame(dictNgrams)
tiidf = tiidf.sort_index()
tiidf = tiidf.fillna(0)
if typeofclass == 2:
    corpus_words = [c for w in corpus_tokens for c in w]
corpus_ngrams = [t for tk in corpus_ngrams for t in tk if t != "Nan"]
# %%
frq_words = pd.DataFrame([" ".join(cn)
                         for cn in corpus_ngrams],
                         columns=['token'])
# for corpus_word in corpus_words:
#     ngram_fd = nltk.FreqDist(ngrams(corpus, n))
frq_words.set_index("token", inplace=True)
# %%
for token, rows in frq_words.iterrows():
    print(rows["corp1"])
# %%
frq_words.to_csv("ngrams.csv")
# %%
for f in filesids:
    print(frq_words[f])
    tokens = {t[f]: i for i, t in frq_words.iterrows()}
    tokens = collections.OrderedDict(sorted(tokens.items(), reverse=True))
    # sortedx = collections.OrderedDict(sorted(tokens.items(), reverse=True))
    x = sorted([v for v in range(len(tokens.items()))], reverse=True)
    y = [k for k, _ in tokens.items()]
    plt.plot(x, y)
    plt.xlabel("Words")
    plt.ylabel("frequency")
plt.show()

# %%
