from re import L
from nltk import ngrams
import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
# typeofclass = int(
#     input("to use word based classification use 1, for character based one use 2"))
# lang = input("enter en for english corpus, ar for the arabic one")

# if lang == "en":
#     lng = "english"
# else:
#     lng = "arabic"
tokenizer = RegexpTokenizer(r'\w+[^a-zA-Z]')
en_stopwords = stopwords.words("english")
nltk.download("punkt")
nltk.download('brown')
nltk.download('stopwords')

# en_corps = """Hello my name is madjid and i would like to try ngrams with zipf law classification,
# and i would like to know how much this will produce and how much this will be fun"""
# %%
# class by word tokenization
en_corps = brown.words()
en_corps = en_corps[:4000]
en_corps = word_tokenize(" ".join(en_corps))
en_corps = tokenizer.tokenize(" ".join(en_corps))
en_corps
# n = int(input("Enter the number of ngrams"))
en_corps = [w for w in en_corps if not w.lower() in en_stopwords]
en_corps = [c for w in en_corps for c in w]
n = 2
n_grams = ngrams(en_corps, n)

# word_fd = nltk.FreqDist(en_corps)
ngram_fd = nltk.FreqDist(ngrams(en_corps, n))


y = sorted(ngram_fd.values(), reverse=True)
x = [i for i in range(len(y))]
plt.loglog(x, y)
plt.xlabel('rank(f)', fontsize=14, fontweight='bold')
plt.ylabel('frequency(r)', fontsize=14, fontweight='bold')
plt.grid(True)
plt.show()
