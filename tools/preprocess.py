# -*- coding: utf-8 -*-

from codecs import open
from sklearn.model_selection import train_test_split
from pycorrector.utils.tokenizer import segment
from tools.config import *

def segment_file(path):
    data_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            source = segment(parts[0].strip(), cut_type='char')
            target = segment(parts[1].strip(), cut_type='char')

            pair = [source, target]
            if pair not in data_list:
                data_list.append(pair)
    return data_list


def _save_data(data_list, data_path):
    with open(data_path, 'w', encoding='utf-8') as f:
        count = 0
        for src, dst in data_list:
            f.write(' '.join(src) + '\t' + ' '.join(dst) + '\n')
            count += 1
        print("save line size:%d to %s" % (count, data_path))


def save_corpus_data(data_list, train_data_path, test_data_path):
    train_lst, test_lst = train_test_split(data_list, test_size=0.1)
    _save_data(train_lst, train_data_path)
    _save_data(test_lst, test_data_path)


if __name__ == '__main__':
    # train data
    data_list = []
    if dataset == 'sighan':
        data_list.extend(segment_file(sighan_train_path))

    save_corpus_data(data_list, train_path, test_path)
