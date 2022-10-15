# %%
import collections
from tracemalloc import start
from nltk import ngrams
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import json
import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk import RegexpTokenizer

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

corpus = PlaintextCorpusReader(
    c_path, ".*", word_tokenizer=RegexpTokenizer('\w+|\$[\d\.]+|\S+|\,'))
corpus = corpus.words()
# corpus = word_tokenize(corpus)

if typeofclass == 2:
    corpus = [c for w in corpus for c in w]
n_grams = ngrams(corpus, n)

ngram_fd = nltk.FreqDist(ngrams(corpus, n))

d_ng = dict(ngram_fd.most_common())
frq_frq = nltk.FreqDist(d_ng.values())
frq_frq = dict(frq_frq.most_common())
zipf_rank = dict()
for i, k in enumerate(d_ng, start=1):
    zipf_rank[i] = [d_ng[k], k]
old_k = 0
zipf_rankk = dict()
cpt = 0
templ = []
frq_frq = dict(sorted(frq_frq.items(), key=lambda item: item[1]))
for i, k in enumerate(frq_frq, start=1):
    print(frq_frq[k])
    templ.append(frq_frq[k])
    if i > 1:
        zipf_rankk[(i-1)+templ[i-2]
                   ] = [kk for kk in list(d_ng.items()) if kk[1] == k]
    else:
        zipf_rankk[i] = [kk for kk in list(d_ng.items()) if kk[1] == k]
f
print([k for k, v in zipf_rankk.items()])
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

# %%
