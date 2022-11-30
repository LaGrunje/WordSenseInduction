import sys
import numpy as np


def fingerprint(text, model, weights=False):
    """
    :param text: list of words
    :param model: word2vec model in Gensim format
    :param weights: whether to use word weights based on frequencies (word2vec model should contain frequency data)
    :return: average vector of words in text
    """
    # Creating list of all words in the document, which are present in the model
    words = [w for w in text if w in model]
    dictionary = list(set(words))
    ln = len(dictionary)
    if ln < 1:
        return np.zeros(model.vector_size)
    vectors = np.zeros((ln, model.vector_size))  # Creating empty matrix of vectors for words
    for i in list(range(ln)):  # Iterate over words in the text
        word = dictionary[i]
        if weights:
            weight = wordweight(word, model)
        else:
            weight = 1.0
        vectors[i, :] = model[word] * weight
    semantic_fingerprint = np.sum(vectors, axis=0)  # Computing sum of all vectors in the document
    semantic_fingerprint = np.divide(semantic_fingerprint, ln)  # Computing average vector
    return semantic_fingerprint


def wordweight(word, model, a=0.001, wcount=250000000):
    """
    :param word: word token
    :param model: word2vec model in Gensim format
    :param a: smoothing coefficient
    :param wcount: number of words in the training corpus (the default value corresponds to the RNC)
    :return: word weight (rare words get higher weights)
    """
    prob = model.wv.vocab[word].count / wcount
    weight = a / (a + prob)
    return weight


def save(df, corpus):
    """
    :param df: Data Frame with predictions
    :param corpus: dataset name
    :return: path to the saved file
    """
    output_file_path = corpus + ".tsv"
    df.to_csv(output_file_path, sep="\t", encoding="utf-8", index=False)
    return output_file_path
