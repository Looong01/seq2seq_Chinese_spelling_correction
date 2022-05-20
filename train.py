# -*- coding: utf-8 -*-

import os
import sys
import importlib

import torch
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.append('../..')

from data_reader import gen_examples
from data_reader import read_vocab, create_dataset, one_hot, save_word_dict, load_word_dict
from seq2seq import Seq2Seq, LanguageModelCriterion
 
config_name, ext = os.path.splitext(os.path.basename(sys.argv[1]))
config = importlib.import_module("config." + config_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: %s' % device)
with open(config.trainlog_path,"w") as f:
    f.write('device: %s' % device)

def evaluate_seq2seq_model(model, data, device, loss_fn, log_path):
    model.eval()
    total_num_words = 0.
    total_loss = 0.
    with torch.no_grad():
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()
            mb_y_len[mb_y_len <= 0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    print("Evaluation loss", total_loss / total_num_words)
    with open(log_path,"a") as f:
        f.write("\nEvaluation loss: {}".format(total_loss / total_num_words))

def train_seq2seq_model(model, train_data, device, loss_fn, optimizer, model_path, epochs, log_path):
    best_loss = 1e3
    train_data, dev_data = train_test_split(train_data, test_size=0.1, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_num_words = 0.
        total_loss = 0.
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(train_data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()
            mb_y_len[mb_y_len <= 0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

            # update optimizer
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            if it % 100 == 0:
                print("Epoch :{}/{}, iteration :{}/{} loss:{:.4f}".format(epoch, epochs, it, len(train_data), loss.item()))
                with open(log_path,"a") as f:
                    f.write("\nEpoch :{}/{}, iteration :{}/{} loss:{:.4f}".format(epoch, epochs, it, len(train_data), loss.item()))

        cur_loss = total_loss / total_num_words
        print("Epoch :{}/{}, Training loss:{:.4f}".format(epoch, epochs, cur_loss))
        with open(log_path,"a") as f:
            f.write("\nEpoch :{}/{}, Training loss:{:.4f}".format(epoch, epochs, cur_loss))

        if epoch % 1 == 0:
            # find best model
            is_best = cur_loss < best_loss
            best_loss = min(cur_loss, best_loss)
            if is_best:
                torch.save(model.state_dict(), model_path)
                print('Epoch:{}, save new bert model:{}'.format(epoch, model_path))
                with open(log_path,"a") as f:
                    f.write('\nEpoch:{}, save new bert model:{}'.format(epoch, model_path))
            if dev_data:
                evaluate_seq2seq_model(model, dev_data, device, loss_fn, log_path)

def train(arch, train_path, batch_size, embed_size, hidden_size, dropout, epochs,
          src_vocab_path, trg_vocab_path, model_path, max_length, log_path):
    source_texts, target_texts = create_dataset(train_path, None)

    src_2_ids = read_vocab(source_texts)
    trg_2_ids = read_vocab(target_texts)
    save_word_dict(src_2_ids, src_vocab_path)
    save_word_dict(trg_2_ids, trg_vocab_path)
    src_2_ids = load_word_dict(src_vocab_path)
    trg_2_ids = load_word_dict(trg_vocab_path)

    id_2_srcs = {v: k for k, v in src_2_ids.items()}
    id_2_trgs = {v: k for k, v in trg_2_ids.items()}
    train_src, train_trg = one_hot(source_texts, target_texts, src_2_ids, trg_2_ids, sort_by_len=True)

    k = 0
    print('src:', ' '.join([id_2_srcs[i] for i in train_src[k]]))
    print('trg:', ' '.join([id_2_trgs[i] for i in train_trg[k]]))

    train_data = gen_examples(train_src, train_trg, batch_size, max_length)
    if arch == 'seq2seq':
        model = Seq2Seq(encoder_vocab_size=len(src_2_ids),
                        decoder_vocab_size=len(trg_2_ids),
                        embed_size=embed_size,
                        enc_hidden_size=hidden_size,
                        dec_hidden_size=hidden_size,
                        dropout=dropout).to(device)
        print(model)
        with open(log_path,"a") as f:
            f.write("\n" + str(model))
        loss_fn = LanguageModelCriterion().to(device)
        optimizer = torch.optim.Adam(model.parameters())

        train_seq2seq_model(model, train_data, device, loss_fn, optimizer, model_path, epochs, log_path)


if __name__ == '__main__':
    train(config.arch,
          config.train_path,
          config.batch_size,
          config.embed_size,
          config.hidden_size,
          config.dropout,
          config.epochs,
          config.src_vocab_path,
          config.trg_vocab_path,
          config.model_path,
          config.max_length,
          config.trainlog_path
          )
