"""
evaluation.py为评测文件
评价指标为Precision、Recall和F1值
predict_path为模型的预测输出的txt文件路径，每一行的样式为: 大 明 紧 张 得 不 得 了 。\n
预测的每一个字用空格分隔，每一句话用换行符分隔。
test_data_path为测试集路径
"""
import pandas as pd

def evaluation(predict_path,test_data_path):
    test_data = [s for i in (pd.read_csv(test_data_path,header=None)).values for s in i]
    predict = [s for i in (pd.read_csv(predict_path,header=None)).values for s in i]
    TP,FP,TN,FN=1e-10,1e-10,1e-10,1e-10

    for id,line in enumerate(test_data):

        w_c = line.split('\t')
        wrong_sentence = w_c[0].split()
        correct_sentence = w_c[1].split()
        if wrong_sentence == correct_sentence:
            if predict[id].split() == correct_sentence:
                TN += 1
            else:
                FN += 1

        if wrong_sentence != correct_sentence:
            if predict[id].split() == correct_sentence:
                TP += 1
            else:
                FP += 1

        # break
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = 2*precision*recall/(precision+recall)

    return TP,FP,TN,FN, precision,recall,F1