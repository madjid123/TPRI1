# %%
from string import punctuation
from nltk import ngrams
from nltk import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import json
import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from collections import Counter
import pandas as pd
# types of params

typeofclass = 1
lang = "en"
n = 1
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


def preprocess_text(text: str):
    # re_tokenizer = RegexpTokenizer(r'\w[^a-zA-Z]')
    tokens = [token for token in nltk.word_tokenize(
        text) if token.lower() not in stoplist]
    return tokens


corpus_ngrams = []
corpus = PlaintextCorpusReader(c_path, ".*")
filesids = corpus._fileids
for f in filesids:
    file_sentences = corpus.sents(f)
    for sent in file_sentences:
        sent_words = preprocess_text(" ".join(sent))
        sent_n_grams = ngrams(sent_words, n)
        corpus_ngrams.append(list(sent_n_grams))

corpus_tokens = preprocess_text(corpus.raw())
# corpus = word_tokenize(corpus)

if typeofclass == 2:
    corpus_words = [c for w in corpus_tokens for c in w]
corpus_tokens = set(corpus_tokens)
corpus_ngrams = [t for tk in corpus_ngrams for t in tk if t != "Nan"]
corpus_ngrams = set([" ".join(cn) for cn in corpus_ngrams])
corpus_ngrams = [cn.split() for cn in corpus_ngrams]
# %%
frq_words = pd.DataFrame([" ".join(cn)
                         for cn in corpus_ngrams],
                         columns=['token'])
# for corpus_word in corpus_words:
#     ngram_fd = nltk.FreqDist(ngrams(corpus, n))
frq_words.set_index("token", inplace=True)
print(frq_words)
for f in filesids:
    frq_words[f] = 0
for crp_ng in corpus_ngrams:
    index = " ".join(crp_ng)
    for f in filesids:
        f_sent = corpus.sents(f)
        crp_gcount = 0
        for sent in f_sent:
            s_tokens = preprocess_text(" ".join(sent))
            s_ngrams = ngrams(s_tokens, n)
            ngrams_counts = Counter(s_ngrams)
            crp_count = ngrams_counts.get(tuple(crp_ng))
            if crp_count != None:
                crp_gcount += crp_count

        frq_words.at[index, f] = crp_gcount
print(frq_words)
# %%
frq_words.to_csv("ngrams.csv")
# %%
