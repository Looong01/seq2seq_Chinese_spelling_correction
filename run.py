# -*- coding: utf-8 -*-

import os
from tools.config import *
from tools.train import train
from tools.infer import Inference
from tools.evaluation import evaluation

batch_size = 8
dropout = 0
epochs = 2000

F1_path = os.path.join(results_path, 'F1.txt')
with open(F1_path, 'w') as f:
    f.truncate(0)

for j in range(6):
    batch_size *= 2
    epochs +=2500
    dropout = 0
    for i in range(20):
        dropout += 0.05
        dropout = round(dropout, 2)
        print(batch_size, dropout, epochs)
        name = 'batch_size-{}_dropout-{}_epochs-{}'.format(batch_size, dropout, epochs)
        model_path = os.path.join(models_path, name + '.pth')
        trainlog_path = os.path.join(trainlogs_path, name + '.log')
        inferlog_path = os.path.join(inferlogs_path, name + '.log')
        predict_path = os.path.join(predicts_path, name + '.txt')
        result_path = os.path.join(results_path, name + '.txt')

        train(arch,
            train_path,
            batch_size,
            embed_size,
            hidden_size,
            dropout,
            epochs,
            src_vocab_path,
            trg_vocab_path,
            model_path,
            max_length,
            trainlog_path
            )

        with open(predict_path,"w") as f:
            f.truncate(0)
        
        m = Inference(arch,
                  model_path,
                  src_vocab_path,
                  trg_vocab_path,
                  embed_size,
                  hidden_size,
                  dropout,
                  max_length
                  )
        src = []
        tgt = []
        fr = open(test_path,'r')
        for line in fr:
            line = line.strip('\n')
            line = line.split('\t')
            src.append(line[0])
            tgt.append(line[1])
        for id, q, in enumerate(src) :
            print('input  :',q)
            print('predict:', m.predict(q))
            print('target: ', tgt[id])
            print()
            with open(inferlog_path,"a") as f:
                f.write('\ninput  :' + q)
                f.write('\npredict:' + m.predict(q))
                f.write('\ntarget: ' + tgt[id])
            with open(predict_path,"a") as f:
                f.write(m.predict(q) + '\n')
        
        TP,FP,TN,FN, precision,recall,F1 = evaluation(predict_path,'data/test.txt')
        print("TP:",TP,'FP:',FP,'TN:',TN,'FN:',FN)
        print('Precision:',precision)
        print('Recall:',recall)
        print('F1:',F1)
        with open(result_path,"a") as f:
            f.write(name + '\n')
            f.write('\nTP: ' + str(TP) + '\tFP: ' + str(FP) + '\tTN: ' + str(TN) + '\tFN: ' + str(FN))
            f.write('\nPrecision: ' + str(precision))
            f.write('\nRecall: ' + str(recall))
            f.write('\nF1: ' + str(F1))
        try:
            os.rename(result_path, '(' + str(F1) + ') ' + result_path)
        except Exception as e:
            print(e)
        with open(F1_path,"a") as f:
            f.write(str(F1) + '\n')