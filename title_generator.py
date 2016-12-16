import chainer
import numpy as np
from chainer import Variable, functions as F, links as L, optimizers, optimizer, serializers, cuda
import os
import pickle

from utils import n_line_iter


GPU_ID = ...
VOCAB_PATH = '../../'
DATA_PATH = ...
N_EPOCH = ...
N_TOPIC = ...
N_HIDDEN = ...
N_EMBED = ...
BATCH_SIZE = ...
MODEL_PATH = 'gen_models'

EOS_TOKEN = '</s>'

v = {}
for word in words:
    if word not in v:
        v[word] = len(v)

if GPU_ID > -1:
    cuda.get_device(GPU_ID).use()
    xp = cuda.cupy
else:
    xp = np


class TitleGenModel(chainer.Chain):
    def __init__(self, n_vocab, n_embed, n_hidden):
        super(TitleGenModel, self).__init__(
            tpc2h=L.Linear(N_TOPIC, n_hidden),
            h2h=L.Linear(n_hidden, 4 * n_hidden),
            y2h=L.EmbedID(n_vocab, 4 * n_hidden),
            h2embed=L.Linear(n_hidden, n_embed),
            embed2y=L.Linear(n_embed, n_vocab)
        )
        self.n_hidden = n_hidden

    def __call__(self, batch_tpc, batch_ts):
        assert len(batch_tpc) == len(batch_ts)
        n_batch = len(batch_tpc)
        n_trg_word = len(batch_ts[0])
        batch_tpc = Variable(batch_tpc)
        lstm_c = self.reset_state(n_batch)
        hs0 = self.tpc2h(batch_tpc)
        lstm_c, hs1 = F.lstm(lstm_c, self.h2h(hs0))
        hyp_titles = [[] for _ in range(n_batch)]
        accum_loss = xp.zeros((), dtype=xp.float32)
        for i in range(n_trg_word):
            embeds = F.tanh(self.h2embed(hs1))
            ys = self.embed2y(embeds)
            ts = Variable(xp.array([batch_ts[k][i] for k in range(n_batch)], dtype=xp.int32))
            accum_loss += F.softmax_cross_entropy(ys, ts)
            hyp_words = ys.data.argmax(1)
            for k in range(n_batch):
                hyp_titles[k].append(hyp_words[k])
            # forward one step
            lstm_c, hs1 = F.lstm(lstm_c, self.h2h(hs1) + self.y2h(ts))
        return hyp_titles, accum_loss

    def decode(self, batch_tpc, max_trg_len=50):
        n_batch = len(batch_tpc)
        batch_tpc = Variable(batch_tpc)
        lstm_c = self.reset_state(n_batch)
        hs0 = self.tpc2h(batch_tpc)
        lstm_c, hs1 = F.lstm(lstm_c, self.h2h(hs0))
        hyp_titles = [[] for _ in range(n_batch)]
        for _ in range(max_trg_len):
            embeds = F.tanh(self.h2embed(hs1))
            ys = self.embed2y(embeds)
            hyp_words = ys.data.argmax(1)
            for k in range(n_batch):
                hyp_titles[k].append(hyp_words[k])
            pred_ts = Variable(hyp_words)
            # forward one step
            lstm_c, hs1 = F.lstm(lstm_c, self.h2h(hs1) + self.y2h(pred_ts))
            # judge whether continuing...
            if all(hyp_titles[k][-1] == vocab[EOS_TOKEN] for k in range(n_batch)):
                break
        return hyp_titles

    def reset_state(n_batch):
        return Variable(np.zeros((n_batch, self.n_hidden)))

    @classmethod
    def load(clf, model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def save(self, model_path):
        with open(model_path, 'wb') as fw:
            pickle.dump(self, fw)


def make_batch(tpcs, ts):
    assert len(tpcs) == len(ts)
    batch_tpc = np.array(tpcs, dtype=xp.float32)
    max_len = max(len(words) for words in ts)
    batch_ts = [words + [vocab[EOS_TOKEN]] * (max_len - len(words) + 1) for words in ts]
    return batch_tpc, batch_ts


if __name__ == '__main__':
    model = TitleGenModel(len(vocab), N_EMBED, N_HIDDEN)

    opt = optimizers.Adam()
    opt.setup(model)
    opt.add_hook(optimizer.GradientClipping(5))

    for epoch in range(N_EPOCH):
        sum_loss = xp.zeros((), dtype=xp.float32)
        for lines in n_line_iter(BATCH_SIZE, DATA_PATH):
            tpcs, ts = ...
            batch_tpc, batch_ts = make_batch(tpcs, ts)
            hyp_titles, loss = model(batch_tpc, batch_ts)
            sum_loss += loss.data
            model.zerograds()
            loss.backward()
            opt.update()
        model.save(os.path.join(MODEL_PATH, 'epoch{}_model.pkl'.format(epoch + 1)))
