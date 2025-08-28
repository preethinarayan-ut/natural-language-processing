# tokenizer.py

import re
from collections import Counter
import nltk

nltk.download("punkt")  # only the first time
nltk.download("punkt_tab")
from nltk.tokenize import word_tokenize

import spacy
nlp = spacy.blank("en")
nlp.max_length = 3_000_000

# Tokenizes a string. Takes a string (a sentence), splits out punctuation and contractions, and returns a list of
# strings, with each string being a token.
def tokenize(string):
    # print(repr(string))
    string = re.sub(r"[^A-Za-z0-9(),.!?\'`\-\"]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\.", " . ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\-", " - ", string)
    string = re.sub(r"\"", " \" ", string)
    # We may have introduced double spaces, so collapse these down
    string = re.sub(r"\s{2,}", " ", string)
    return list(filter(lambda x: len(x) > 0, string.split(" ")))

def top_words_whitespace(filename, top_n=10):
    """
    Reads a text file, tokenizes by whitespace only, and returns the top N words.
    """
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # Whitespace-based tokenization (no punctuation stripping)
    tokens = text.split()

    # Count word frequencies
    counter = Counter(tokens)

    # Return the top N most frequent words
    return counter.most_common(top_n)

def top_words_custom(filename, top_n=10):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Custom tokenization using the provided tokenize function
    tokens = tokenize(text)
    counter = Counter(tokens)
    return counter.most_common(top_n)

def top_words_nltk(filename, top_n=10):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    
    # NLTK tokenization
    tokens = word_tokenize(text)
    counter = Counter(tokens)
    return counter.most_common(top_n)


def top_words_spacy(filename, top_n=10):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    
    # spaCy tokenization
    doc = nlp(text)
    tokens = [token.text for token in doc]
    counter = Counter(tokens)
    return counter.most_common(top_n)

if __name__ == "__main__":
    print(repr(tokenize("said.")))
    print(repr(tokenize("said?\"")))
    print(repr(tokenize("I didn't want to, but I said \"yes!\" anyway.")))

    filename = "nyt.txt"

    print("Top words with whitespace tokenization:")
    print(top_words_whitespace(filename, top_n=10))

    print("Top words with custom tokenization:")
    print(top_words_custom(filename, top_n=10))

    print("Top words with NLTK tokenization:")
    print(top_words_nltk(filename, top_n=10))

    print("Top words with spaCy tokenization:")
    print(top_words_spacy(filename, top_n=10))


    