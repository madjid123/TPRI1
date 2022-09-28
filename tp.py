from nltk import ngrams
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import json
import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader

# types of params
typeofclass = 2
lang = "en"
n = 2
c_path = "./corpora"
c_path = os.path.abspath(c_path)
typeofclass = int(
    input("to use word based classification use 1, for character based one use 2: "))
lang = input("enter en for english corpus, ar for the arabic one: ")

if lang == "en":
    c_path += "/english"
else:
    c_path += "/arabic"

corpus = PlaintextCorpusReader(c_path, ".*")
corpus = corpus.words()
# corpus = word_tokenize(corpus)

n = int(input("Enter the number of ngrams: "))
if typeofclass == 2:
    corpus = [c for w in corpus for c in w]
n_grams = ngrams(corpus, n)

ngram_fd = nltk.FreqDist(ngrams(corpus, n))

y = sorted(ngram_fd.values(), reverse=True)
x = [i for i in range(len(y))]

plt.loglog(x, y)
plt.xlabel('rank(f)', fontsize=14, fontweight='bold')
plt.ylabel('frequency(r)', fontsize=14, fontweight='bold')
plt.grid(True)
plt.savefig("zipflaw.pdf")
plt.show()
f = open("classement.txt", "w", encoding="utf-8")
f.write(json.dumps(ngram_fd.most_common(), ensure_ascii=False))
f.close()
