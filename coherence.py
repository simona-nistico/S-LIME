import re
import numpy as np
import pandas as pd
import pickle 

#import spacy
#import spacy_sentence_bert
from sentence_transformers import SentenceTransformer, util

from functools import partial
from sklearn.feature_extraction.text import CountVectorizer
from text_interpretable import get_neighs, NeigborhoodIndexedStrings, LimeTextExplainerLatent, LemmaTokenizer
from lime.lime_text import LimeTextExplainer, IndexedString
from sklearn.tree import DecisionTreeClassifier

def distance_fn(sample, samples):
    """
    Distance computation, we consider each sentence as a binary bag of word, the aim of this measure is to give a
    greater distance measure to sentences with much words different to the original one
    Args:
        sample: reference sample
        samples: samples for which we want to compute distance
    Returns:
        distances
    """

    splitter = re.compile(r'(%s)|$' % '\W+')
    non_word = splitter.match
    # We are assuming that words are yet tokenized
    s_w = set([s for s in splitter.split(sample) if s and not non_word(s)])
    distances = np.empty(len(samples))
    for i in range(len(samples)):
        distances[i] = len(set([s for s in splitter.split(samples[i]) if s and not non_word(s)]).difference(s_w))
    return distances

def distance_fn2(tf, sample, samples):
  path = ""
  #spacy.prefer_gpu()
  #tf = spacy.load('en_use_md') # custom medium
  distances = np.empty(len(samples))
  embedding1 = tf.encode(sample, convert_to_tensor=True)
  for i in range(len(samples)):
    embedding2 = tf.encode(samples[i], convert_to_tensor=True)
    #distances[i] = embd_sample.similarity(tf(samples[i]))#*len(samples[i])
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    distances[i] = abs(cosine_scores) # AGGIUNTO -> RENDO POS  
  return distances

def compute_explanations(lte, tlt, dataset, classifier_fn, encode, decode, lt, rho=25):
    """
    Compute explanations for LIME and S-LIME on a given dataset
    Args:
        lte: LimeTextExplainer
        tlt: LimeTextExplainerLatent
        dataset: data for which we want to produce explanations, list with shape (n_samples,)
        classifier_fn: classification function for black-box model to explain
        encode: function to encode samples in latent space
        decode: function  to decode samples from latent space
        lt: LemmaTokenizer
        rho: int, radius to consider for generation
    Returns:
        trees, indexes, explanations, indexes_lime
    """

    def kernel(d, sigma):
      return np.sqrt(np.exp(-(d ** 2) / sigma ** 2))

    def kernel_custom(d):
      return np.exp(d)-1

    kernel_fn = partial(kernel_custom)#partial(kernel, sigma=5)

    trees = np.empty((len(dataset)), dtype=np.object)
    indexes = np.empty((len(dataset)), dtype=np.object)
    explanations = np.empty((len(dataset)), dtype=np.object)
    indexes_lime = np.empty((len(dataset)), dtype=np.object)

    tf = SentenceTransformer('stsb-roberta-large')

    semantic_values = []

    for i in range(len(dataset)):
        # Compute Standard Lime Explanation
        #e = lte.explain_instance(dataset[i], classifier_fn, num_features=4)

        # Compute explanation using latent neighborhood and decision tree
        z = encode(dataset[i])
        Z = get_neighs(z, rho, 400)
        I = decode(Z)
        I = np.append([dataset[i]], I, axis=0)
        I_l = lt(I)
        ind = NeigborhoodIndexedStrings(I_l)
        data, yss, _ = tlt.data_labels_distances(ind, classifier_fn)
        distances = distance_fn2(tf, I_l[0], I_l)
        weights = kernel_fn(distances)

        # Convert probability in labels
        if len(yss.shape) > 1:
            yss = np.argmax(yss, axis=1)
        
        tree = DecisionTreeClassifier(max_depth=20)
        tree.fit(data, yss, weights)
        trees[i] = tree
        indexes[i] = ind

        semantic_values.append( distances )

        # Extract words for lime
        #ins = IndexedString(dataset[i])

        #explanations[i] = e
        #indexes_lime[i] = ins

        if (i+1) % 100 == 0:
            print(i+1,'processed')

    return trees, indexes, explanations, indexes_lime, semantic_values


def compare_explanations(trees, indexes, explanations, indexes_lime, lt): # correzione in data 19-04-2021 del commento returns per farlo combaciare con il primo pezzo
    """
    Compare explanations obtained by LIME and by S-LIME producing tree sets
        pl_at - words that are in lime explanation but not in S-LIME explanation
        al_pt - words the are in S-LIME explanation but no in LIME explanation
        pl_pt - words that are both in LIME and S-LIME explanation
    Args:
        trees: list of DecisionTreeClassifier (num_samples, )
        indexes: list of NeighborhoodIndexedString (num_samples, )
        explanations: list of Explanations (num_samples, )
        indexes_lime: list of IndexedString (num_samples, )
        lt: LemmaTokenizer
    Returns
        pl_at: list of sets, words that are in lime explanation but not in S-LIME explanation
        al_pt: list of sets, words the are in S-LIME explanation but no in LIME explanation
        pl_pt: list of sets, words that are both in LIME and S-LIME explanation
    """
    pl_at = np.empty((len(trees), ), dtype=np.object)
    pl_pt = np.empty((len(trees), ), dtype=np.object)
    al_pt = np.empty((len(trees), ), dtype=np.object)
    slime_res = np.empty((len(trees), ), dtype=np.object)
    for i in range(len(trees)):

        tree = trees[i]
        ind = indexes[i]

        # Extract words for tree model
        features = tree.tree_.feature[tree.tree_.feature >= 0]
        words_tlt = set(pd.Series(features).apply(lambda f: ind.inverse_vocab[f]))

        # Extract words for lime
        #e = explanations[i]
        #ins = indexes_lime[i]
        #words_lime = set(pd.Series(np.array(e.local_exp[1])[:, 0], dtype=int).apply(
        #    lambda f: lt([ins.inverse_vocab[f]])[0]))

        words_lime = set(pd.Series())

        pl_pt[i] = words_tlt.intersection(words_lime).difference(set(['']))
        pl_at[i] = words_lime.difference(words_tlt).difference(set(['']))
        al_pt[i] = words_tlt.difference(words_lime).difference(set(['']))
        slime_res[i] = words_tlt

        if (i+1) % 100 == 0:
            print(i+1, ' processed')

    return pl_at, pl_pt, al_pt, slime_res


def compute_score(sum_pos, sum_tot):
    """
    Compute the score defined in section 5 starting the count of term occurrences in sentences with positive label and
    the total occurrences (each term could have two values in each sentence, 1 if the term is present at least one time,
    0 otherwise)
    Args:
        sum_pos: numpy array (1, num_words), term occurrences in sentences with positive label
        sum_tot: numpy array (1, num_words), total terms occurrences
    Returns:
        score: numpy array (num_words, )
    """
    k = np.where(sum_tot - sum_pos > sum_pos, sum_tot - sum_pos, sum_pos) + 1
    n = sum_tot + 2

    score_infos = np.array(
        (1 - k / n,
         (1 - k / n) + (n * k / n * (1 - k / n)) ** 0.5 / n,
         k / n - (k * (1 - k / n)) ** 0.5 / n,
         k / n))
    score_infos = score_infos.transpose()
    #score_infos = np.squeeze(score_infos) ## aggiunta per evitare errore nella linea successiva


    score = np.where(score_infos[:, 1] > score_infos[:, 2], 0, (k / n - (k * (1 - k / n)) ** 0.5 / n - 0.5) / 0.5)
    return score
