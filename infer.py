# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import torch

sys.path.append('../..')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import config
from data_reader import SOS_TOKEN, EOS_TOKEN
from data_reader import load_word_dict
from seq2seq import Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: %s' % device)


class Inference(object):
    def __init__(self, arch, model_path, src_vocab_path, trg_vocab_path,
                 embed_size=50, hidden_size=50, dropout=0.5, max_length=128):
        self.src_2_ids = load_word_dict(src_vocab_path)
        self.trg_2_ids = load_word_dict(trg_vocab_path)
        self.id_2_trgs = {v: k for k, v in self.trg_2_ids.items()}
        if arch == 'seq2seq':
            self.model = Seq2Seq(encoder_vocab_size=len(self.src_2_ids),
                                 decoder_vocab_size=len(self.trg_2_ids),
                                 embed_size=embed_size,
                                 enc_hidden_size=hidden_size,
                                 dec_hidden_size=hidden_size,
                                 dropout=dropout).to(device)

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.arch = arch
        self.max_length = max_length

    def predict(self, query):
        result = []
        tokens = [token.lower() for token in query]
        tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
        src_ids = [self.src_2_ids[i] for i in tokens if i in self.src_2_ids]

        if self.arch == 'seq2seq':
            src_tensor = torch.from_numpy(np.array(src_ids).reshape(1, -1)).long().to(device)
            src_tensor_len = torch.from_numpy(np.array([len(src_ids)])).long().to(device)
            sos_tensor = torch.Tensor([[self.trg_2_ids[SOS_TOKEN]]]).long().to(device)
            translation, attn = self.model.translate(src_tensor, src_tensor_len, sos_tensor, self.max_length)
            translation = [self.id_2_trgs[i] for i in translation.data.cpu().numpy().reshape(-1) if i in self.id_2_trgs]

        for word in translation:
            if word != EOS_TOKEN:
                result.append(word)
            else:
                break
        return ''.join(result)


if __name__ == "__main__":
    m = Inference(config.arch,
                  config.model_path,
                  config.src_vocab_path,
                  config.trg_vocab_path,
                  embed_size=config.embed_size,
                  hidden_size=config.hidden_size,
                  dropout=config.dropout,
                  max_length=config.max_length
                  )

    test_inputs = pd.read_csv('data/test.tsv',sep='\t')
    src = [x for x in test_inputs['#src']]
    tgt = [x for x in test_inputs['target']]
    for id, q, in enumerate(src) :
        print('input  :',q)
        print('predict:', m.predict(q))
        print('target: ', tgt[id])
        print()

