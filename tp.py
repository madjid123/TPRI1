from nltk import ngrams

sentence = 'this is a foo bar sentences and i want to ngramize it'

n = 6
n_grams = ngrams(sentence.split(), n)

for grams in n_grams:
    print(grams)
