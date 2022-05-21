# -*- coding: utf-8 -*-

import os

batch_size = 128
dropout = 0.25
epochs = 200

max_length = 128
gpu_id = 1
embed_size = 128
hidden_size = 128

pwd_path = os.path.abspath(os.path.dirname(__file__) + '/../')

# Training data path.

data_dir = os.path.join(pwd_path, 'data')

trainlogs_path = os.path.join(pwd_path, 'output/logs/train')
inferlogs_path = os.path.join(pwd_path, 'output/logs/infer')
models_path = os.path.join(pwd_path, 'output/models')
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
model_path = os.path.join(models_path, 'batch_size-{}_dropout-{}_epochs-{}.pth'.format(batch_size, dropout, epochs))
trainlog_path = os.path.join(trainlogs_path, 'batch_size-{}_dropout-{}_epochs-{}.log'.format(batch_size, dropout, epochs))
inferlog_path = os.path.join(inferlogs_path, 'batch_size-{}_dropout-{}_epochs-{}.log'.format(batch_size, dropout, epochs))

if not os.path.exists(trainlogs_path):
    os.makedirs(trainlogs_path)
if not os.path.exists(inferlogs_path):
    os.makedirs(inferlogs_path)
if not os.path.exists(models_path):
    os.makedirs(models_path)