from math import ceil

import torch
import numpy as np
import os
import pickle
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib_venn_wordcloud import venn2_wordcloud

from Bag_VAE import Bag_VAE, text_process
from autoencoders.model import VAE, DAE, AAE, reparameterize
from autoencoders.vocab import Vocab
from autoencoders.noise import noisy
from autoencoders.utils import set_seed, strip_eos
from autoencoders.batchify import get_batches
from classifier.classifier_nn import build_nn
from nltk.corpus import stopwords
import lime.lime_text
from text_interpretable import *
from coherence import compute_explanations, compare_explanations, compute_score_fabrizio


def encode(sents):
    sents_conv = pd.Series(sents).apply(lambda x: x.split())
    batches, order = get_batches(sents_conv, vocab, 256, device)
    z = []
    for inputs, _ in batches:
        mu, logvar = model.encode(inputs)
        zi = reparameterize(mu, logvar)
        z.append(zi.detach().cpu().numpy())
    z = np.concatenate(z, axis=0)
    z_ = np.zeros_like(z)
    z_[np.array(order)] = z
    return z_


def decode(z, batch_size=512):
    sents = []
    for i in range(ceil(len(z)/batch_size)):
        zi = torch.tensor(z[i*batch_size: min((i+1)*batch_size, len(z))], device=device)
        outputs = model.generate(zi, 35, 'greedy').t()
        results = pd.Series(outputs[:, 1:].cpu()).apply(lambda x: [vocab.idx2word[id] for id in x])
        sents = np.append(sents, results, axis=0)
    return pd.Series(strip_eos(sents)).apply(lambda x: ' '.join(x)).tolist()


def get_model(path):
    ckpt = torch.load(path)
    train_args = ckpt['args']
    model = {'dae': DAE, 'vae': VAE, 'aae': AAE}[train_args.model_type](
        vocab, train_args).to(device)
    model.load_state_dict(ckpt['model'])
    model.flatten()
    model.eval()
    return model

# def encode(sents):
#     enc_sents = model.vocab.transform(sents)
#     mean, logvar = model.encode(enc_sents.toarray())
#     z = model.reparameterize     (mean, logvar)
#     return z
#
#
# def decode(z):
#     bow = model.sample(z)
#     bow = np.where(bow > 0.5, 1, 0)
#     words = model.vocab.inverse_transform(bow)
#     sents = pd.Series(words).apply(lambda x: ' '.join(x))
#     return sents
#
# def get_model(path, model):
#     ckpt = tf.train.Checkpoint(step=tf.Variable(1, name="step"), net=model)
#     manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=1000)
#     ckpt.restore(manager.latest_checkpoint).assert_consumed()


def print_lime_latent_explanation(i, set1, set2, sents, lt):
    f, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, facecolor='#ffddb7', figsize=(6.4, 4))
    f.set_rasterized(True)
    ax1.axis(False)
    ax1.text(0.3, 1, 'Classification: ' + str(np.where(classifier_fn([sents[i]])[0, 0] == 0, 'Negative', 'Positive')),
             horizontalalignment='center', color='#355c7d', fontsize=14)
    ax1.text(0.5, 0.5, 'SENTENCE: \n' + sents[i] + '\n LEMMATIZATION: \n[' + lt([sents[i]])[0] + ']', wrap=True,
             horizontalalignment='center', verticalalignment='center')
    venn2_wordcloud([set1[i], set2[i]], set_labels=["LIME", "XXX"],
                    set_edgecolors=['#584848', '#f3b54a'], ax=ax0)
    plt.savefig('images\explanation_{}.eps'.format(i), facecolor='#ffddb7')
    plt.show()


def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            sents.append(line)
    return sents


if __name__ == '__main__':
    data = 'yelp'
    if data == 'yelp':
        path = 'checkpoints/yelp/daae'
        # path = 'bag_VAE/checkpoints1_64'
        dataset = 'data/yelp/sentiment'
        class_folder = 'classifier/yelp'
        # data = pd.read_csv(dataset)

        # Exctract data of interest
        # data_classes = data[(data['stars'] == 1) | (data['stars'] == 5)]
        # x = data_classes['text']
        # y = data_classes['stars'].where(data_classes['stars'] == 1, other=2) - 1
        sents = load_sent(os.path.join(dataset, '1000.pos'))
        n_pos = len(sents)
        sents.extend(load_sent(os.path.join(dataset, '1000.neg')))
        y = np.ones(len(sents), dtype=int)
        y[n_pos:] = 0

        vocab = Vocab(os.path.join(path, 'vocab.txt'))
        # set_seed(1111)
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        model = get_model(os.path.join(path, 'model.pt'))

        vec = pickle.load(open(os.path.join(class_folder, 'vec_nn.pickle'), 'rb'))
        classifier = build_nn(len(vec.vocabulary_))
        classifier.load_weights(os.path.join(class_folder, 'model_dense.hdf5'))
        classifier_fn = lambda s: classifier.predict(vec.transform(s).toarray())

        cv_d = pickle.load(open('tests/tree/model_2/cv_d.pickle', 'rb'))
        score_fabrizio_correzione = pickle.load(open('tests/tree/model_2/score_fabrizio_distance_2.pickle', 'rb'))

    if data == 'yahoo':
        path = 'checkpoints/yahoo/daae'
        class_folder = 'classifier/yahoo'
        dataset = 'data/yahoo_answers_csv/'
        sents = pickle.load(open(os.path.join(dataset, 'test_2_processed.pickle'), 'rb'))
        y = pickle.load(open(os.path.join(dataset, 'test_2_processed_y.pickle'), 'rb'))

        vocab = Vocab(os.path.join(path, 'vocab.txt'))
        # set_seed(1111)
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        model = get_model(os.path.join(path, 'model.pt'))

        mnb = pickle.load(open(os.path.join(class_folder, 'mnb.pickle'), 'rb'))
        vec = pickle.load(open(os.path.join(class_folder, 'vec.pickle'), 'rb'))
        classifier_fn = lambda s: mnb.predict_proba(vec.transform(s))

        cv_d = pickle.load(open('tests/yahoo/cv_d.pickle', 'rb'))
        score_fabrizio_correzione = pickle.load(open('tests/yahoo/score_fabrizio_correzione.pickle', 'rb'))


    # Load classifier
    # classifier = pickle.load(open('models/mnb.pickle', 'rb'))
    # vocab_classifier = pickle.load(open(os.path.join('models', 'vocab.pickle'), 'rb'))
    # classifier_fn = lambda s: classifier.predict_proba(vocab_classifier.transform(s))

    score_dict = dict(zip(cv_d.get_feature_names(), score_fabrizio_correzione))

    def score_parola(w):
        if w in score_dict:
            return score_dict[w]
        return 0.0

    stop_words = pickle.load(open('stop_words.pickle', 'rb'))
    lt = LemmaTokenizer(split_expression=r'\W+', stop_words=stop_words)
    tlt = LimeTextExplainerLatent()
    lte = lime.lime_text.LimeTextExplainer()
    exit()

    # ----------------------- Compute explanations ---------------------------------------
    print('----------- COMPUTE EXPLANATIONS ----------------')
    trees, indexes, explanations, indexes_lime = compute_explanations(lte, tlt, sents, classifier_fn, encode, decode,
                                                                      lt)
    print('------------------- DONE ------------------------')

    # ----------------------- Compute indexes --------------------------------------------
    print('---------------- COMPUTE INDEXES -----------------')
    pl_at, pl_pt, al_pt = compare_explanations(trees, indexes, explanations, indexes_lime, lt)
    tree_words = al_pt.copy()
    for i in range(len(sents)):
        tree_words[i] = tree_words[i].union(pl_pt[i])
    lime_words = pl_at.copy()
    for i in range(len(sents)):
        lime_words[i] = lime_words[i].union(pl_pt[i])

    tree_words_score = pd.Series(tree_words).apply(lambda x: [score_parola(w) for w in x])
    lime_words_score = pd.Series(lime_words).apply(lambda x: [score_parola(w) for w in x])
    tree_words_top_4 = tree_words.copy()
    for i in range(len(sents)):
        tree_words_top_4[i] = np.array(list(tree_words[i]))[np.argsort(-np.array(tree_words_score[i]))[:4]]
    tree_words_top_4_score = pd.Series(tree_words).apply(lambda x: [score_parola(w) for w in x])
    tree_words_top_4_score_s = pd.Series(tree_words_top_4_score).apply(np.sum)
    lime_words_score_s = pd.Series(lime_words_score).apply(np.sum)
    tree_words_top_4_score_m = tree_words_top_4_score_s / 4
    lime_words_score_m = lime_words_score_s / 4

    al_pt_top_4 = tree_words.copy()
    pl_pt_top_4 = tree_words.copy()
    pl_at_top_4 = tree_words.copy()
    tree_words_top_4 = pd.Series(tree_words_top_4).apply(set)
    for i in range(len(sents)):
        al_pt_top_4[i] = tree_words_top_4[i].difference(lime_words[i])
        pl_pt_top_4[i] = lime_words[i].intersection(tree_words_top_4[i])
        pl_at_top_4[i] = lime_words[i].difference(tree_words_top_4[i])

    lime_words_diff = pd.Series(lime_words).apply(len).apply(lambda x: 4 - x)
    al_pt_top_4_score = pd.Series(al_pt_top_4).apply(lambda x: [score_parola(w) for w in x])
    pl_pt_top_4_score = pd.Series(pl_pt_top_4).apply(lambda x: [score_parola(w) for w in x])
    pl_at_top_4_score = pd.Series(pl_at_top_4).apply(lambda x: [score_parola(w) for w in x])
    al_pt_top_4_score_s = al_pt_top_4_score.apply(np.sum)
    pl_pt_top_4_score_s = pl_pt_top_4_score.apply(np.sum)
    pl_at_top_4_score_s = pl_at_top_4_score.apply(np.sum)
    al_pt_top_4_score_m = al_pt_top_4_score.apply(np.mean)
    pl_pt_top_4_score_m = pl_pt_top_4_score.apply(np.mean)
    pl_at_top_4_score_m = pl_at_top_4_score_s / (pd.Series(pl_at_top_4).apply(len) + lime_words_diff)
    print('------------------- DONE ------------------------')

    # cv_d = pickle.load(open('tests/tree/model_2/cv_d.pickle', 'rb'))
    # score_fabrizio_correzione = pickle.load(open('tests/tree/model_2/score_fabrizio_distance_2.pickle', 'rb'))
