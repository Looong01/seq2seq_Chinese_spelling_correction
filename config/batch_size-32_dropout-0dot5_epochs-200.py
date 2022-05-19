# -*- coding: utf-8 -*-

import os

batch_size = 32
dropout = 0.5
epochs = 200

max_length = 128
gpu_id = 1
embed_size = 128
hidden_size = 128

pwd_path = os.path.abspath(os.path.dirname(__file__) + '/../')

# Training data path.

sighan_train_path = os.path.join(pwd_path, 'data/train.tsv')

output_dir = os.path.join(pwd_path, 'output/batch_size-{}_dropout-{}_epochs-{}'.format(batch_size, dropout, epochs))
# Training data path.
train_path = os.path.join(pwd_path, 'data/train.txt')
# Validation data path.
test_path = os.path.join(pwd_path, 'data/valid.txt')

dataset = 'sighan'
arch = 'seq2seq'

# config
src_vocab_path = os.path.join(pwd_path, 'data/vocab_source.txt')
trg_vocab_path = os.path.join(pwd_path, 'data/vocab_target.txt')
model_path = os.path.join(output_dir, 'model_batch_size-{}_dropout-{}_epochs-{}.pth'.format(batch_size, dropout, epochs))
log_path = os.path.join(output_dir, 'model_batch_size-{}_dropout-{}_epochs-{}.log'.format(batch_size, dropout, epochs))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
