import numpy as np
import pdb
import re
import matplotlib.pyplot as plt

fp = open("./print_mAP.txt", "r")

lines = fp.readlines()

#y = [re.findall(r'-?\d+\.?\d*e?[-+]?\d*?', item) for item in lines]
y = [re.findall(r'\d+\.?\d*e?[-+]?\d+', item) for item in lines]
y = [[int(item[0]), eval(item[1])] for item in y]

ny = []

for i in range(200):
    id = i + 1
    ny.append(sum([item[1] for item in y if item[0] == id]))


x = np.arange(200)

plt.figure()
plt.xlabel('epoch')
plt.ylabel('accuracy')
#plt.title('Loss curve of neural network training model for garbage classification')

plt.plot(x, ny)

plt.savefig("./accuracy.jpg")

plt.show()