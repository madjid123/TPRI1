from nltk import ngrams
import nltk
en_corps = """Hello my name is madjid and i would like to try ngrams with zipf law classification, 
and i would like to know how much this will produce and how much this will be fun"""
n = int(input("Enter the number of ngrams"))
n_grams = ngrams(en_corps.split(), n)
for ng in n_grams:
    print(ng)
    freq = nltk.FreqDist(ng)
    print(freq.values())
