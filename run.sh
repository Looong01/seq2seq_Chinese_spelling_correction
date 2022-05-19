source /f2020/yang/anaconda3/bin/activate
conda activate PyTorch

# 实验1
python train.py batch_size-32_dropout-0dot25_epochs-200.py
python train.py batch_size-64_dropout-0dot25_epochs-200.py
python train.py batch_size-128_dropout-0dot25_epochs-200.py
python train.py batch_size-256_dropout-0dot25_epochs-200.py

# 实验2
python train.py batch_size-32_dropout-0dot5_epochs-200.py
python train.py batch_size-64_dropout-0dot5_epochs-200.py
python train.py batch_size-128_dropout-0dot5_epochs-200.py
python train.py batch_size-256_dropout-0dot5_epochs-200.py

# 实验3


# 实验4