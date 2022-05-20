# -*- coding: utf-8 -*-

import os

pwd_path = os.path.abspath(os.path.dirname(__file__))


data_dir = os.path.join(pwd_path, 'data')
output_dir = os.path.join(pwd_path, 'output')
# Training data path.
train_path = os.path.join(data_dir, 'train.txt')
# Validation data path.
valid_path = os.path.join(data_dir, 'valid.txt')
# Validation data path.
test_path = os.path.join(data_dir, 'test.txt')

arch = 'seq2seq'
# config
src_vocab_path = os.path.join(data_dir, 'vocab_source.txt')
trg_vocab_path = os.path.join(data_dir, 'vocab_target.txt')
model_path = os.path.join(output_dir, 'model_{}.pth'.format(arch))


batch_size = 32
epochs = 200
max_length = 128
gpu_id = 0
dropout = 0.25
embed_size = 128
hidden_size = 128

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
