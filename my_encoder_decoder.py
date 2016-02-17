# -*- coding: utf-8 -*-
__author__ = 'mana-ysh'

'''
MEMO

小田さんの1epochごとにoptimizerを初期化してるけどいいのか?
シャッフルは前処理でやっとく?
処理時間測定する関数作る
batchサイズより小さくなった残り文の学習
ビーム探索の実装および単語の出力を確率分布に従って出力するよう実装
ArgumentParser使う
語順を逆にするのどうする？
コーパスをクリーニングして文の数を減らす？

'''


import datetime
import chainer.functions as F
import numpy as np
from chainer import Variable, FunctionSet, optimizers
import pickle
from collections import defaultdict
import argparse

class EncoderDecoderModel:
    def __init__(self, src_vocab, trg_vocab, n_embed=256, n_hidden=512, algorithm='Adam'):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.algorithm = algorithm
        self.model = FunctionSet(embed_x=F.EmbedID(len(src_vocab), n_embed),
                                 en_x_to_h=F.Linear(n_embed, 4*n_hidden),
                                 en_h_to_h=F.Linear(n_hidden, 4*n_hidden),
                                 en_h_to_de_h=F.Linear(n_hidden, 4*n_hidden),
                                 de_h_to_embed_y=F.Linear(n_hidden, n_embed),
                                 embed_y_to_y=F.Linear(n_embed, len(trg_vocab)),
                                 y_to_h=F.EmbedID(len(trg_vocab), 4*n_hidden),
                                 de_h_to_h=F.Linear(n_hidden, 4*n_hidden))

    def get_model(self):
        return self.model

    def forward(self, src_batch, trg_batch):
        # encode
        n_batch = len(src_batch)
        lstm_c = self.initialize_state(n_batch)
        src_sent_words = len(src_batch[0])
        for i in range(src_sent_words):
            print np.array([src_batch[k][i] for k in range(n_batch)], dtype=np.int32)
            x = Variable(np.array([src_batch[k][i] for k in range(n_batch)], dtype=np.int32))
            en_x = F.tanh(self.model.embed_x(x))
            if i == 0:
                lstm_c, en_h = F.lstm(lstm_c, self.model.en_x_to_h(en_x))
            else:
                lstm_c, en_h = F.lstm(lstm_c, self.model.en_x_to_h(en_x) + self.model.en_h_to_h(en_h))

        # decode
        hyp_sents = [[] for i in range(n_batch)]
        accum_loss = Variable(np.zeros(()).astype(np.float32))
        trg_sent_words = len(trg_batch[0])
        lstm_c, de_h = F.lstm(lstm_c, self.model.en_h_to_de_h(en_h))
        for i in range(trg_sent_words):
            embed_y = F.tanh(self.model.de_h_to_embed_y(de_h))
            y = self.model.embed_y_to_y(embed_y)
            t = Variable(np.array([trg_batch[k][i] for k in range(n_batch)], dtype=np.int32))
            accum_loss += F.softmax_cross_entropy(y, t)
            output = y.data.argmax(1)
            for k in range(n_batch):
                hyp_sents[k].append(output[k])
            lstm_c, de_h = F.lstm(lstm_c, self.model.de_h_to_h(de_h) + self.model.y_to_h(t))
        return hyp_sents, accum_loss

    def fit(self, src_batch, trg_batch):
        self.optimizer.zero_grads()
        hyp_sents, accum_loss = self.forward(src_batch, trg_batch)
        accum_loss.backward()
        self.optimizer.clip_grads(10)
        self.optimizer.update()
        return hyp_sents

    def predict(self, src_batch, sent_len_limit):
        # encode
        n_batch = len(src_batch)
        lstm_c = self.initialize_state(n_batch)
        src_sent_words = len(src_batch[0])
        for i in range(src_sent_words):
            x = Variable(np.array([src_batch[k][i] for k in range(n_batch)], dtype=np.int32))
            en_x = F.tanh(self.model.embed_x(x))
            if i == 0:
                lstm_c, en_h = F.lstm(lstm_c, self.model.en_x_to_h(en_x))
            else:
                lstm_c, en_h = F.lstm(lstm_c, self.model.en_x_to_h(en_x) + self.model.en_h_to_h(en_h))

        # decode
        lstm_c, de_h = F.lstm(lstm_c, self.model.en_h_to_de_h(en_h))
        hyp_sents = [[] for i in range(n_batch)]

        # output the highest probability words
        while len(hyp_sents[0]) < sent_len_limit:
            embed_y = F.tanh(self.model.de_h_to_embed_y(de_h))
            y = self.model.embed_y_to_y(embed_y)
            output = y.data.argmax(1)
            for k in range(n_batch):
                hyp_sents[k].append(output[k])
            output = Variable(output)
            lstm_c, de_h = F.lstm(lstm_c, self.model.de_h_to_h(de_h) + self.model.y_to_h(output))
            if all(hyp_sents[k][-1] == trg_vocab['</s>'] for k in range(n_batch)):
                break
        return hyp_sents

    def initialize_optimizer(self, lr=0.5):
        if self.algorithm == 'SGD':
            self.optimizer = optimizers.SGD(lr=lr)
        elif self.algorithm == 'Adam':
            self.optimizer = optimizers.Adam()
        elif self.algorithm == 'Adagrad':
            self.optimizer = optimizers.AdaGrad()
        elif self.algorithm == 'Adadelta':
            self.optimizer = optimizers.AdaDelta()
        else:
            raise AssertionError('this algorithm is not available')
        self.optimizer.setup(self.model)

    def initialize_state(self, n_batch):
        return Variable(np.zeros((n_batch, self.n_hidden), dtype=np.float32))

def make_vocab(filename, n_vocab=False):
    vocab = {}
    sents = []
    with open(filename, 'r') as f:
        if n_vocab is False:
            for sent in f.readlines():
                sent_words = sent.split()
                for i, word in enumerate(sent_words):
                    if word not in vocab:
                        vocab[word] = len(vocab)
                    sent_words[i] = vocab[word]
                sents.append(sent_words)
            return np.array(sents), vocab
        else:
            n_worddict = defaultdict(lambda: 0)
            raw_sents = f.readlines()
            for sent in raw_sents:
                for word in sent.split():
                    n_worddict[word] += 1
            cut_dict = dict(sorted(n_worddict.items(), key=lambda x: x[1], reverse=True)[:n_vocab-1])
            vocab = {k:v for v,k in enumerate(cut_dict.keys())}
            vocab['<unk>'] = len(vocab)
            for sent in raw_sents:
                sent_words = sent.split()
                for i, word in enumerate(sent_words):
                    if word not in vocab:
                        sent_words[i] = vocab['<unk>']
                    else:
                        sent_words[i] = vocab[word]
                sents.append(sent_words)
            return np.array(sents), vocab

def make_batch(batch_sents, end_token_id):
    max_len = max(len(sent) for sent in batch_sents)
    return [sent + [end_token_id] * (max_len - len(sent)) for sent in batch_sents]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', default=20, type=int, help='the number of epochs to learn')
    parser.add_argument('--batchsize', '-b', default=20, type=int, help='the number of mini-batchsize')
    parser.add_argument('--embed', '-em', default=256, type=int, help='embedding layer size')
    parser.add_argument('--hidden', '-hi', default=512, type=int, help='hidden layer size')
    parser.add_argument('--vocab_cutting', '-v', default=False, type=int, help='the number of vocabs using in model')
    parser.add_argument('--algorithm', '-a', choices=['SGD', 'Adagrad', 'Adam', 'Adadelta'], default='Adam',
                        help='what kind of al')
    args = parser.parse_args()
    n_epoch = args.epoch
    batchsize = args.batchsize
    n_embed = args.embed
    n_hidden = args.hidden
    n_vocab = args.vocab_cutting
    algorithm = args.algorithm
    print '%d epoch\n' % n_epoch
    print '%d minibatch-size\n' % batchsize
    print '%d embedding layer\n' % n_embed
    print '%d hidden layer\n' % n_hidden
    print 'using %s algorithm to optimize\n' % algorithm
    if n_vocab is False:
        print 'not cutting vocab\n'
    else:
        print '%d vocabs\n' % n_vocab
    src_data, src_vocab = make_vocab('../data/normal2.small', n_vocab=n_vocab)
    trg_data, trg_vocab = make_vocab('../data/simple2.small', n_vocab=n_vocab)
    print 'src : %d vocabs' % len(src_vocab)
    print 'trg : %d vocabs\n' % len(trg_vocab)
    inv_src_vocab = {v : k for k, v in src_vocab.items()}
    inv_trg_vocab = {v : k for k, v in trg_vocab.items()}
    if not len(src_data) == len(trg_data):
        raise AssertionError('the number of sentences is different')
    n_sent = len(src_data)
    jump = int(n_sent / batchsize)
    EDmodel = EncoderDecoderModel(src_vocab=src_vocab, trg_vocab=trg_vocab, n_embed=n_embed, n_hidden=n_hidden, algorithm=algorithm)
    EDmodel.initialize_optimizer()
    n_trained = 0
    for epoch in range(n_epoch):
        print datetime.datetime.now(), 'start %d epoch' % (epoch + 1)
        #EDmodel.initialize_optimizer()
        for i in range(jump):
            src_batch = make_batch(src_data[(i * batchsize):(i * batchsize + batchsize)], src_vocab['</s>'])
            trg_batch = make_batch(trg_data[(i * batchsize):(i * batchsize + batchsize)], trg_vocab['</s>'])
            hyp_batch = EDmodel.fit(src_batch, trg_batch)
            # output src, trg, hyp sentences
            for k in range(batchsize):
                print datetime.datetime.now(), '%d epoch, %dth sentence' % (epoch + 1, n_trained + k + 1)
                print 'src =' + ' '.join([inv_src_vocab[x] for x in src_batch[k]])
                print 'trg =' + ' '.join([inv_trg_vocab[x] for x in trg_batch[k]])
                print 'hyp =' + ' '.join([inv_trg_vocab[x] for x in hyp_batch[k]]) + '\n'
            n_trained += batchsize
        print datetime.datetime.now(), 'finished %d epoch' % (epoch + 1), 'and dumping model\n'
        pickle.dump(EDmodel.get_model(), open('./epoch%d.model' % (epoch + 1), 'wb'))
