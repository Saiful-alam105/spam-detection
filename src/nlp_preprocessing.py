import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize once
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def lowercase_text(text):
    return text.lower()


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_special_characters(text):
    return re.sub(r"[^a-zA-Z\s]","",text)


def tokenize_text(text):
    return word_tokenize(text)


def remove_stopwords(tokens):
    return [
        word
        for word in tokens
        if word not in stop_words
    ]


def stem_words(tokens):
    return [
        stemmer.stem(word)
        for word in tokens
    ]


def preprocess_text(text):

    text = lowercase_text(text)

    text = remove_punctuation(text)

    text = remove_special_characters(text)

    tokens = tokenize_text(text)

    tokens = remove_stopwords(tokens)

    tokens = stem_words(tokens)

    return " ".join(tokens)