# sentiment_data.py

from typing import List


class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.
    Represents a single sentence as a list of words and a label (0 or 1).

    Attributes:
        words (List[string]): list of words (sentence split into words)
        label (int): 0 or 1 (0 = negative, 1 = positive)
    """

    def __init__(self, words, label):
        """
        :param words: list of words (List[string])
        :param label: 0 or 1 (0 = negative, 1 = positive)
        """
        self.words = words
        self.label = label

    def __repr__(self):
        """
        :return: Returns a string representation of the SentimentExample
        """
        return repr(self.words) + "; label=" + repr(self.label)

    def __str__(self):
        """
        :return: Returns a string representation of the SentimentExample
        """
        return self.__repr__()


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and cleans the sentences and forms
    SentimentExamples.

    Reads a file of labeled sentiment examples, one per line, with the label (0 or 1) followed by a tab, and turns them into a list of SentimentExamples.

    File format:
    0    I hate this movie
    1    This movie was great

    Steps inside:
    - open the file
    - go through each line
    - if the line isn't empty:
        - split the line into label (0 or 1) and sentence on the tab character
        - if there is no tab character, split on whitespace and take the first field as the label and the rest as the sentence
        - convert the label to an integer (0 or 1)
        - tokenize the sentence by splitting on spaces (split the sentence into words)
        - clean the tokenized sentence by removing empty tokens
        - store it as a SentimentExample with the tokenized cleaned sentence and the label


    :param infile: file to read from
    :return: a list of SentimentExamples parsed from the file
    """
    f = open(infile)
    exs = []
    for line in f:
        if len(line.strip()) > 0:
            fields = line.split("\t")
            if len(fields) != 2:
                fields = line.split()
                label = 0 if "0" in fields[0] else 1
                sent = " ".join(fields[1:])
            else:
                # Slightly more robust to reading bad output than int(fields[0])
                label = 0 if "0" in fields[0] else 1
                sent = fields[1]
            tokenized_cleaned_sent = list(filter(lambda x: x != '', sent.rstrip().split(" ")))
            exs.append(SentimentExample(tokenized_cleaned_sent, label))
    f.close()
    return exs


def read_blind_sst_examples(infile: str) -> List[List[str]]:
    """
    Reads the blind SST test set, which just consists of unlabeled sentences
    
    For reading test data without labels. Each line is a single sentence.

    Example file structure:
    I really liked the movie
    This movie was terrible

    Splits each sentence into words by splitting on spaces.

    :param infile: path to the file to read
    :return: list of tokenized sentences (list of list of strings)
    """
    f = open(infile, encoding='utf-8')
    exs = []
    for line in f:
        if len(line.strip()) > 0:
            exs.append(line.split(" "))
    return exs


def write_sentiment_examples(exs: List[SentimentExample], outfile: str):
    """
    Writes sentiment examples to an output file with one example per line, the predicted label followed by the example.
    Note that what gets written out is tokenized.

    This function writes out a list of SentimentExamples to a file, one per line, in the format:
    [0 or 1]<TAB>[tokenized sentence]
    (Writes examples back to a file in the format they were read in, but tokenized.)

    Example input:
    exs = [SentimentExample(["I", "love", "this", "movie"], 1), SentimentExample(["This", "movie", "is", "bad"], 0)]

    Example output:
    1    I love this movie
    0    This movie is bad

    :param exs: the list of SentimentExamples to write
    :param outfile: out path
    :return: None
    """
    o = open(outfile, 'w')
    for ex in exs:
        o.write(repr(ex.label) + "\t" + " ".join([word for word in ex.words]) + "\n")
    o.close()

