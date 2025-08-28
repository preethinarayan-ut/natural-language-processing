import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

def get_smart_tokens(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = word_tokenize(text)
    # Filter: only alphabetic tokens, lowercase, and not stopwords
    stops = set(stopwords.words("english"))
    tokens = [w.lower() for w in tokens if w.isalpha() and w.lower() not in stops]
    return tokens

def plot_zipf(filename, top_n=500):
    tokens = get_smart_tokens(filename)
    counter = Counter(tokens)
    most_common = counter.most_common(top_n)

    # Rank = 1, 2, 3, ...
    ranks = list(range(1, len(most_common) + 1))
    freqs = [count for _, count in most_common]
    inv_ranks = [1.0 / r for r in ranks]

    plt.figure(figsize=(8,6))
    plt.plot(inv_ranks, freqs, marker="o", linestyle="none", alpha=0.6)
    plt.xlabel("Inverse Rank (1 / rank)")
    plt.ylabel("Word Frequency")
    plt.title("Zipf’s Law: Inverse Rank vs Word Frequency (Smart Tokenization)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_zipf("nyt.txt")
