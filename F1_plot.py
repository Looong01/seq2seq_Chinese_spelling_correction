# -*- coding: utf-8 -*-

import os
from tools.config import *
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 30), dpi=300)
ax = fig.add_subplot(311,projection='3d')
ax1 = fig.add_subplot(312,projection='3d')
ax2 = fig.add_subplot(313,projection='3d')
x = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
y = np.array([16, 32, 64, 128, 256, 512])
dropout, batchsize = np.meshgrid(x, y)
F1_path = os.path.join(results_path, 'F1.txt')
with open(F1_path,'r') as f:
    datas = f.readlines()
F1 = np.zeros((6, 19), dtype=float)
k = 0
for i in range(6):
    for j in range(19):
        F1[i, j] = datas[k]
        k+=1
print('MAX is')
print('F1: ', F1.max())
index = np.argwhere(F1 == F1.max())
for i in range(len(index)):
    print((i+1), '.  Dropout: ', dropout[index[i][0]][index[i][1]], '\tBatch Size: ', batchsize[index[i][0]][index[i][1]])
ax.set_xlabel('Dropout')
ax.set_ylabel('Batch Size')
ax.set_zlabel('F1')
ax.plot_wireframe(dropout, batchsize, F1, color='red')
ax.set_title('Normal Degree')
ax1.set_xlabel('Dropout')
ax1.set_ylabel('Batch Size')
ax1.set_zlabel('F1')
ax1.plot_wireframe(dropout, batchsize, F1, color='red')
ax1.view_init(elev=0, azim=0)
ax1.set_title('Batch Size - F1')
ax2.set_xlabel('Dropout')
ax2.set_ylabel('Batch Size')
ax2.set_zlabel('F1')
ax2.plot_wireframe(dropout, batchsize, F1, color='red')
ax2.view_init(elev=0, azim=90)
ax2.set_title('Dropout - F1')
plt.savefig('F1.png')