import numpy as np
import pdb
import re
import matplotlib.pyplot as plt

fp = open("./print_mAP.txt", "r")

lines = fp.readlines()
#r"\d+\.?\d*"
#r'-?\d+\.?\d*e?-?\d*?'
y = [re.findall(r"\d+\.?\d*", item) for item in lines]
y = [[int(item[0]), eval(item[1])] for item in y]

ny = []

for i in range(199):
    id = i + 1
    ny.append(sum([item[1] for item in y if item[0] == id]))


x = np.arange(199)

plt.figure()
plt.xlabel('epoch')
plt.ylabel('accuracy')
#plt.title('map curve of neural network training model for 8-stackedHourglass')

plt.plot(x, ny)

plt.savefig("./map.jpg")

plt.show()