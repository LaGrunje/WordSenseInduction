from os import path
from pandas import read_csv
from evaluate import evaluate
import argparse
import sys
import numpy as np
import gensim
import logging
from sklearn.cluster import AffinityPropagation, SpectralClustering

sys.path.append("./utils/")
from utils import save, fingerprint

sys.path.append("./preprocess/")
from preprocess import preprocess

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--input', help='Path to input file with contexts', required=True)
    arg('--model', help='Path to word2vec model', required=True)
    arg('--weights', dest='weights', action='store_true', help='If it is neccessary to use weights of the model')
    arg('--2stage', dest='twostage', action='store_true', help='Use 2-stage clustering')
    arg('--test', dest='testing', action='store_true', help='Work with test csv files')

    parser.set_defaults(testing=False)
    parser.set_defaults(twostage=False)
    parser.set_defaults(weights=False)
    args = parser.parse_args()

    modelfile = args.model
    if modelfile.endswith('.bin.gz'):  # Word2vec binary format
        model = gensim.models.KeyedVectors.load_word2vec_format(modelfile, binary=True)
    elif modelfile.endswith('.vec.gz'):  # Word2vec text format
        model = gensim.models.KeyedVectors.load_word2vec_format(modelfile, binary=False)
    else:  # word2vec in Gensim native format
        model = gensim.models.KeyedVectors.load(modelfile)
    model.init_sims(replace=True)
    dataset = args.input

    damping = 0.8
    preference = -0.9

    df = read_csv(dataset, sep="\t", encoding="utf-8")
    res = df.copy()
    df = preprocess(df)

    predicted = []
    goldsenses = []
    for query in df.word.unique():
        subset = df[df.word == query]
        if not args.testing:
            goldsenses.append(len(subset.gold_sense_id.unique()))
        contexts = []
        matrix = np.empty((subset.shape[0], model.vector_size))
        counter = 0
        lengths = []
        for line in subset.iterrows():
            con = line[1].context
            identifier = line[1].context_id
            label = query + str(identifier)
            contexts.append(label)
            if type(con) == float:
                fp = np.zeros(model.vector_size)
            else:
                bow = con.split()
                bow = [b for b in bow if b != query]
                fp = fingerprint(bow, model, weights=args.weights)
                lengths.append(len(bow))
            matrix[counter, :] = fp
            counter += 1
        clustering = AffinityPropagation(preference=preference, damping=damping, random_state=None).fit(matrix)

        if args.twostage:
            nclusters = len(clustering.cluster_centers_indices_)
            if nclusters < 1:
                nclusters = 1
            elif nclusters == len(contexts):
                nclusters = 4
            clustering = SpectralClustering(n_clusters=nclusters, n_init=20,
                                            assign_labels='discretize', n_jobs=2).fit(matrix)
        cur_predicted = clustering.labels_.tolist()
        predicted += cur_predicted
        if not args.testing:
            gold = subset.gold_sense_id

    res.predict_sense_id = predicted
    fname = path.splitext(path.basename(args.input))[0]
    if args.testing:
        save(res, fname)
    else:
        res = evaluate(save(res, fname))
        print('ARI:', res)
        print('Average number of senses:', np.average(goldsenses))
        print('Variation of the number of senses:', np.std(goldsenses))
        print('Minimum number of senses:', np.min(goldsenses))
        print('Maximum number of senses:', np.max(goldsenses))


if __name__ == '__main__':
    main()
