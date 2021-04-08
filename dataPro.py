# coding: utf-8

import pandas as pd
import numpy as np
import pdb

fp = open('./valid.txt', 'w')

df_fi = pd.read_excel('./fileToId.xlsx')
df_id = pd.read_excel('./idToData.xlsx')


data_fi = df_fi.values
data_id = df_id.values

imgs = data_fi[:,0]
ids = data_fi[:,1]
oths = []

for sid in ids:
    find = False
    for item in data_id:
        if sid == item[0]:
            oths.append([item[1], item[3], item[6], item[7]])
            find = True
    if find == False:
        oths.append([' ', ' ', ' ', ' '])

results = []
for i, (img, oth) in enumerate(zip(imgs, oths)):
    result = []
    result.append(img)
    for item in oth:
        result.append(item)
    results.append(result)


for item in results:
    if not item[0].endswith('jpg') and not item[0].endswith('png'):
        if not item[2] == 'Unverified':
            print(item[0] + ' ' + item[2])

'''
maxd = 0
mind = 99999
for item in results:
    mon = item[1].year * 12 + item[1].month
    if mon < mind:
        mind = mon
    if mon > maxd:
        maxd = mon

posResults = []
nevResults = []
for item in results:
    if not item[0].endswith('jpg') and not item[0].endswith('png') or item[2] == 'Unverified':
        continue
    if item[2] == 'Positive ID':
        posResults.append(item)
    else:
        nevResults.append(item)

results = []
for item in posResults:
    for i in range(50):
        results.append(item)
nevResults = np.array(nevResults)
nevResults = nevResults[-701:]
for item in nevResults:
    results.append(item)

for item in results:
    fp.write('data/images/'+item[0]+'$+$')
    fp.write(str((maxd - (item[1].year * 12 + item[1].month)) / (maxd - mind))+'$+$')
    fp.write(str(item[3] / 90.0)+'$+$')
    fp.write(str(item[4] / 180.0)+'$+$')
    if item[2] == 'Positive ID':
        fp.write('1\n')
    else:
        fp.write('0\n')

fp.flush()
fp.close()
'''