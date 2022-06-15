# Seq2Seq中文拼写纠错

#### Description
Web_Search_Tech期末大作业

#### Software Architecture
*  Chinese typo correction model based on equence to Sequence  

This model is essentially two recurrent neural networks, the encoder is used to analyze the input sequence and the decoder is used to generate the output sequence.

The role of the encoder is to transform an indefinitely long input sequence into a fixed-length background variable **c**, and encode the input sequence information in the background variable.

The decoder gets the next hidden state based on the background vector **c** of the encoder output, the predicted output, and the previous hidden layer state, and with the hidden state of the decoder, we can use the custom output layer and softmax operations to calculate the probability distribution of the predictor word.


#### Installation

* Use `pip` to install the dependencies
```
conda create -n Web_Search_Tech python=3.9
conda activate Web_Search_Tech
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install scikit-learn pandas matplotlib
pip install pycorrector
```

#### Instructions

* train
```
cd <the path of this project>
conda activate Web_Search_Tech
train.sh
```

* infer
```
cd <the path of this project>
conda activate Web_Search_Tech
infer.sh
```

#### Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request


#### Gitee Feature

1.  You can use Readme\_XXX.md to support different languages, such as Readme\_en.md, Readme\_zh.md
2.  Gitee blog [blog.gitee.com](https://blog.gitee.com)
3.  Explore open source project [https://gitee.com/explore](https://gitee.com/explore)
4.  The most valuable open source project [GVP](https://gitee.com/gvp)
5.  The manual of Gitee [https://gitee.com/help](https://gitee.com/help)
6.  The most popular members  [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
