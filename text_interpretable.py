import itertools
import re
from functools import partial

import lime
from lime.lime_text import IndexedString, TextDomainMapper
import numpy as np
import scipy as sp
import sklearn
import nltk
from nltk.corpus import wordnet
from nltk import WordNetLemmatizer, word_tokenize
from sklearn.utils import check_random_state
from sklearn.feature_extraction.text import CountVectorizer


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


class LemmaTokenizer(object):
    def __init__(self, split_expression, stop_words):
        self.wnl = WordNetLemmatizer()
        self.splitter = re.compile(r'(%s)|$' % split_expression)
        self.stop_words = stop_words

    def __call__(self, articles, stop_words=[]):
        result = []
        for r in articles:
            tmp = [self.wnl.lemmatize(t.lower(), get_wordnet_pos(t.lower())) for t in self.splitter.split(r)
                           if t and (t.lower() not in self.stop_words) and (not self.splitter.match(t))]
            result.append(' '.join(tmp))
        return result


def get_neighs(z, rho, n_samples, sigma_min=0):
    r = np.empty((n_samples, z.shape[1]), dtype=np.float32)
    r[:, :] = np.random.uniform(sigma_min, rho, size=(n_samples, 1))
    theta = np.zeros((n_samples, z.shape[1]))
    theta[:, 0] = 2 * np.pi
    theta[:, 1:z.shape[1] - 1] = np.random.uniform(0, np.pi, size=(n_samples, z.shape[1] - 2))
    theta[:, z.shape[1] - 1] = np.random.uniform(0, 2 * np.pi, size=n_samples)
    si = np.sin(theta)
    si[:, 0] = 1
    si = np.cumprod(si, axis=1)
    co = np.cos(theta)
    co = np.roll(co, -1, axis=1)
    Z = np.empty((n_samples, z.shape[1]))
    Z[:, :] = z
    return (Z + (si * co * r)).astype(np.float32)


def generate_data(expression, encode_fn, decode_fn, n_samples, rho):
    # Encode string
    z = encode_fn([expression])
    # Produce neighborhood
    Z = get_neighs(z, rho, n_samples-1)
    E = np.append(expression, decode_fn(Z))
    return Z, E


class NeigborhoodIndexedStrings(IndexedString):
    def __init__(self, raw_strings, split_expression=r'\W+', bow=True,
                 mask_string=None):
        """Initializer.
        Args:
            raw_string: string with raw text in it
            split_expression: Regex string or callable. If regex string, will be used with re.split.
                If callable, the function should return a list of tokens.
            bow: if True, a word is the same everywhere in the text - i.e. we
                 will index multiple occurrences of the same word. If False,
                 order matters, so that the same word will have different ids
                 according to position.
            mask_string: If not None, replace words with this if bow=False
                if None, default value is UNKWORDZ
        """
        super(NeigborhoodIndexedStrings, self).__init__(raw_strings[0].lower(), split_expression=split_expression,
                                                        bow=bow, mask_string=mask_string)
        self.raw = raw_strings
        self.mask_string = 'UNKWORDZ' if mask_string is None else mask_string
        self.split_expression = split_expression

        if callable(split_expression):
            tokens = split_expression(self.raw)
            self.as_list = self._segment_with_tokens(self.raw, tokens)
            tokens = set(tokens)

            def non_word(string):
                return string not in tokens

        else:
            splitter = re.compile(r'(%s)|$' % split_expression)
            non_word = splitter.match
            for i in range(1, len(self.raw)):
                # with the split_expression as a non-capturing group (?:), we don't need to filter out
                # the separator character from the split results.
                self.as_list.extend([s for s in splitter.split(self.raw[i].lower()) if s])

        self.as_np = np.array(self.as_list)
        self.string_start = np.hstack(
            ([0], np.cumsum([len(x) for x in self.as_np[:-1]])))
        self.vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        non_vocab = set()
        for i, word in enumerate(self.as_np):
            if word in non_vocab:
                continue
            if non_word(word):
                non_vocab.add(word)
                continue
            if bow:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                    self.inverse_vocab.append(word)
                    self.positions.append([])
                idx_word = self.vocab[word]
                self.positions[idx_word].append(i)
            else:
                self.inverse_vocab.append(word)
                self.positions.append(i)
        if not bow:
            self.positions = np.array(self.positions)

    def raw_string(self):
        """Returns the original raw string"""
        return self.raw[0]


class LimeTextExplainerLatent():
    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 split_expression=r'\W+',
                 bow=True,
                 mask_string=None,
                 random_state=None,
                 char_level=False):
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.base = lime.lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.mask_string = mask_string
        self.split_expression = split_expression
        self.char_level = char_level

    # TODO sistemare gli argomenti
    def __data_labels_distances(self,
                                indexed_string,
                                classifier_fn,
                                # num_samples,
                                distance_metric='cosine',
                                stop_words=[]):
        """Generates a neighborhood around a prediction.
        Generates neighborhood data by randomly removing words from
        the instance, and predicting with the classifier. Uses cosine distance
        to compute distances between original and perturbed instances.
        Args:
            indexed_strings: document (IndexedString) to be explained and
                generated strings,
            classifier_fn: classifier prediction probability function, which
                takes a string and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity.
        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of tokens in indexed_string. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: cosine distance between the original instance and
                    each perturbed instance (computed in the binary 'data'
                    matrix), times 100.
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100

        doc_size = indexed_string.num_words()
        # lt = LemmaTokenizer(indexed_string.split_expression, stop_words)
        # vocab = CountVectorizer()
        # data = vocab.fit_transform(lt(indexed_string.raw))
        # Build bag of word representation (like the original lime work)
        splitter = re.compile(r'(%s)|$' % indexed_string.split_expression)
        data = np.zeros((len(indexed_string.raw), doc_size))
        for i in range(len(indexed_string.raw)):
            indexes = [indexed_string.vocab[s] for s in splitter.split(indexed_string.raw[i].lower())
                       if not (s==None or splitter.match(s))]
            data[i][indexes] = 1

        labels = classifier_fn(indexed_string.raw)
        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances

    def explain(self, x, encode_fn, decode_fn, class_fun, rho, n_samples=100, labels=(1,), num_features=10,
                model_regressor=None, stop_words=None):
        _, E = generate_data(x, encode_fn, decode_fn, n_samples, rho)
        indexer = NeigborhoodIndexedStrings(E)
        domain_mapper = TextDomainMapper(indexer)
        data, yss, distances = self.__data_labels_distances(indexer, class_fun)

        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = lime.explanation.Explanation(domain_mapper=domain_mapper, class_names=self.class_names,
                                               random_state=self.random_state)
        ret_exp.predict_proba = yss[0]
        ret_exp.score = {}
        ret_exp.local_pred = {}
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

#     def get_neights_hidden(self, indexed_string, num_samples):
#         splitter = re.compile(r'(%s)|$' % indexed_string.split_expression)
#         indexes = [indexed_string.vocab[s] for s in splitter.split(indexed_string.raw[0].lower())
#                    if not (s == None or splitter.match(s))]
#
#         doc_size = indexed_string.indexed_string.num_words()
#         sample = self.random_state.randint(1, doc_size + 1, num_samples - 1)
#         data = np.ones((num_samples, doc_size))
#         data[0][indexes] = 1
#         features_range = range(doc_size)
#         inverse_data = [indexed_string.indexed_string.raw_string()]
#         for i, size in enumerate(sample, start=1):
#             inactive = self.random_state.choice(features_range, size,
#                                                 replace=False)
#             data[i, inactive] = 0
#             inverse_data.append(indexed_string.indexed_string.inverse_removing(inactive))
#         return data, inverse_data