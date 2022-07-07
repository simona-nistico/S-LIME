import os
import argparse
import pickle
import string
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Input, Reshape, Conv1D, Flatten
from tensorflow.python.keras import Sequential

from BagGenerator import BagGenerator
from utils import logging, load_sent


class Bag_VAE(tf.keras.Model):
    def __init__(self, vocab, vocab_len, latent_dim=64, units_enc=[512, 256, 128], units_dec=[128, 256, 512],
                 d_rate=0.3):
        super(Bag_VAE, self).__init__()
        self.vocab = vocab
        self.vocab_len = vocab_len
        self.latent_dim = latent_dim

        self.encoder = Sequential()
        self.encoder.add(Input(shape=(vocab_len, )))
        for u in units_enc:
            self.encoder.add(Dense(u, activation=tf.nn.selu, kernel_initializer=tf.keras.initializers.he_normal(),
                                   kernel_regularizer=tf.keras.regularizers.l1()))
            self.encoder.add(BatchNormalization())
            self.encoder.add(Dropout(d_rate))
        self.encoder.add(Dense(2 * latent_dim))

        self.decoder = Sequential()
        self.decoder.add(Input(shape=(latent_dim,)))
        for u in units_dec:
            self.decoder.add(Dense(u, activation=tf.nn.selu, kernel_initializer=tf.keras.initializers.he_normal()))
            self.decoder.add(BatchNormalization())
            self.decoder.add(Dropout(d_rate))
        self.decoder.add(Dense(vocab_len))

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x, pos_weight=2):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    logits = model.decode(z)
    #reconstr_loss = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(x, logits, pos_weight=pos_weight), 1)
    reconstr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits), 1)
    # reconstr_loss = tf.reduce_sum(w * tf.square(x - logits), 1)
    kdl = - tf.reduce_sum(1 + 2 * logvar - tf.square(mean) - tf.exp(2 * logvar), 1)
    return tf.reduce_mean(reconstr_loss + kdl)


# def compute_loss(model, x, pos_weight=2):
#     mean, logvar = model.encode(x)
#     z = model.reparameterize(mean, logvar)
#     logits = model.decode(z)
#
#     cross_entr = tf.nn.weighted_cross_entropy_with_logits(labels=x, logits=logits, pos_weight=pos_weight)
#     logpx_z = -tf.reduce_sum(cross_entr, axis=1)
#     logpz = 2 * log_normal_pdf(z, 0., 0.)
#     logqz_x = 2 * log_normal_pdf(z, mean, logvar)
#     return - tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer, loss_fc, weights):
    """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
    with tf.GradientTape() as tape:
        loss = loss_fc(model, x, weights)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def text_process(text):
    # Data cleaning function
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word.lower() for word in nopunc.split()] # if word.lower() not in set(stopwords.words('english'))]


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', metavar='FILE', default='data/yelp',
                    help='path to training file')
parser.add_argument('--save-dir', default='bag_VAE/binary_weighted_128', metavar='DIR',
                    help='directory to save checkpoints and outputs')
parser.add_argument('--load-model', default='', metavar='FILE',
                    help='path to load checkpoint if specified')
parser.add_argument('--dim_z', type=int, default=64, metavar='D',
                    help='dimension of latent variable z')
parser.add_argument('--dropout', type=float, default=0.4, metavar='DROP',
                    help='dropout probability (0 = no dropout)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='batch size')

if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log_file = os.path.join(args.save_dir, 'log.txt')
    logging(str(args), log_file)


    # Exctract data of interest
    # data = pd.read_csv(os.path.join(args.dataset, 'yelp.csv'))
    # data_classes = data[(data['stars'] == 1) | (data['stars'] == 5)]# Prepare data
    # x = data_classes['text']
    # y = data_classes['stars'].where(data_classes['stars'] == 1, other=2) - 1
    # train_sents, valid_sents, y_train, y_valid = train_test_split(x, y, test_size=0.1, random_state=45)
    train_sents = load_sent(os.path.join(args.dataset, 'train.txt'))
    logging('# train sents {}, tokens {}'.format(
        len(train_sents), sum(len(s) for s in train_sents)), log_file)
    valid_sents = load_sent(os.path.join(args.dataset, 'valid.txt'))
    logging('# valid sents {}, tokens {}'.format(
        len(valid_sents), sum(len(s) for s in valid_sents)), log_file)

    vocab_file = os.path.join(args.save_dir, 'vocab.pickle')
    train_file = os.path.join(args.save_dir, 'proc_train.pickle')
    valid_file = os.path.join(args.save_dir, 'proc_test.pickle')
    if not os.path.isfile(vocab_file):
        vocab = CountVectorizer(analyzer=text_process, min_df=3).fit(train_sents)
        print(len(vocab.vocabulary_))
        pickle.dump(vocab, open(vocab_file, 'wb'))
        # Prepare dataset
        proc_data_train = vocab.transform(train_sents)
        proc_data_valid = vocab.transform(valid_sents)
        pickle.dump(proc_data_train, open(train_file, 'wb'))
        pickle.dump(proc_data_valid, open(valid_file, 'wb'))
    else:
        print('Loading vocab')
        vocab = pickle.load(open(vocab_file, 'rb'))
        print('Vocab loaded')
        print('Loading data')
        proc_data_train = pickle.load(open(train_file, 'rb'))
        proc_data_valid = pickle.load(open(valid_file, 'rb'))
        print('Data loaded')

    bg_train = BagGenerator(proc_data_train, batch_size=args.batch_size)
    bg_valid = BagGenerator(proc_data_valid, batch_size=args.batch_size)

    units_enc = [512, 256]
    units_dec = [256, 512]
    #units_enc = [512, 256, 128]
    #units_dec = [128, 256, 512]
    logging('----- Structure -----', log_file)
    logging(str(units_enc), log_file)
    logging(str(units_dec), log_file)
    logging('-------- End --------', log_file)

    # model = Bag_VAE(vocab=vocab, vocab_len=len(vocab.vocabulary_), latent_dim=args.dim_z, units_enc=[512, 256, 128],
    #                 units_dec=[128, 256, 512], d_rate=args.dropout)
    model = Bag_VAE(vocab=vocab, vocab_len=len(vocab.vocabulary_), latent_dim=args.dim_z, units_enc=units_enc,
                    units_dec=units_dec, d_rate=args.dropout)
    print(model.encoder.summary())
    print(model.decoder.summary())

    # optimizer = tf.keras.optimizers.Adagrad()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)


    step = tf.Variable(1, name="step")
    ckpt = tf.train.Checkpoint(step=step, net=model)
    manager = tf.train.CheckpointManager(ckpt, args.save_dir, max_to_keep=1000)

    # Compute weights (words doesn't have the same frequencies, we want to set an higher weight for
    # less frequent words)
    # w = np.sum(proc_data_train.toarray(), axis=0)
    # #w = proc_data_train.shape[0] / (w * len(vocab.vocabulary_))
    # w = ((w - w.min()) / w.max()).astype('float32')

    pos_weight = 1.3
    logging('Pos weights: '+str(pos_weight), log_file)

    # Train
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        loss = tf.keras.metrics.Mean()
        for i in range(len(bg_train)):
            train_x, _ = bg_train.__getitem__(i)
            batch_loss = train_step(model, train_x, optimizer, compute_loss, pos_weight)
            loss(batch_loss)
        end_time = time.time()
        elbo_tr = -loss.result()

        loss = tf.keras.metrics.Mean()
        for i in range(len(bg_valid)):
            valid_x, _ = bg_valid.__getitem__(i)
            loss(compute_loss(model, valid_x, pos_weight))
        elbo = -loss.result()

        # if epoch % 10 == 0:
        #     save_path = manager.save()
        #
        #     symbolic_weights = getattr(optimizer, 'weights')
        #     weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
        #     with open(args.save_dir+'/optimizer_{}.pkl'.format(epoch), 'wb') as f:
        #         pickle.dump(weight_values, f)
        #
        #     print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        log_output = '|| Epoch: {} | train :{} | test: {} | time elapsed: {} ||'.format(epoch, elbo_tr, elbo,
                                                                                        end_time - start_time)
        logging(log_output, log_file)
        step.assign_add(1)