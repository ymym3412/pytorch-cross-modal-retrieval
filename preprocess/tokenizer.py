import nltk
from nltk import tokenize


class RootTokenizer():
    def tokenize(self, sentence):
        raise NotImplementedError


class EnglishTokenizer(RootTokenizer):
    def __init__(self):
        nltk.download('punkt')

    def tokenize(self, sentence):
        return tokenize.word_tokenize(sentence)

