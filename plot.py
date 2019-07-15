# coding: utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt

with open(sys.argv[1]) as f:
    lines = f.read().splitlines()
    
loss = []
acc = []
val_loss = []
val_acc = []

for line in lines:
    l = float(line.split(': ')[1].split(' ')[0])
    a = float(line.split(': ')[2].split(' ')[0])
    vl = float(line.split(': ')[3].split(' ')[0])
    va = float(line.split(': ')[4])
    loss.append(l)
    acc.append(a)
    val_loss.append(vl)
    val_acc.append(va)
    
x = np.arange(len(loss))

plt.figure()
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x, loss, label='train loss')
plt.plot(x, val_loss, label='val loss')
plt.legend()
plt.show()

plt.figure()
plt.title('accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(x, acc, label='train accuracy')
plt.plot(x, val_acc, label='val accuracy')
plt.legend()
plt.show()
