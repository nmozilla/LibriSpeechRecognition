import pandas as pd
import numpy as np

#orig = pd.read_csv('../LibriSpeech/train.csv', delimiter=',')
c = 0
len_i = []

with open('../dutch_clips_flac/train.csv', 'r') as infile:
    with open('../DutchDataset/dutch-train-small.csv','w') as f:
        f.write('idx,input,label\n')
        for i in infile.readlines():
            len_i.append(len(i))
            if len(i) < 290 and len(i) > 260:
                i = str(c) + i[i.find(','):]
                f.write(i)
                c += 1
print(str(c) + ' lines copied in train!')

#print('len_i',len_i)
#print('min', min(len_i))
#print('max', max(len_i))
#print('mean', min(len_i)+max(len_i)/2)





d = 0

with open('../dutch_clips_flac/dev.csv', 'r') as infile:
    with open('../DutchDataset/dutch-dev-small.csv','w') as f:
        f.write('idx,input,label\n')
        for i in infile.readlines():
            if len(i) < 290 and len(i) > 260:
                i = str(d) + i[i.find(','):]   
                f.write(i)
                d += 1
print(str(d) + ' lines copied in dev!')

b = 0
with open('../dutch_clips_flac/test.csv', 'r') as infile:
    with open('../DutchDataset/dutch-test-small.csv','w') as f:
        f.write('idx,input,label\n')
        for i in infile.readlines():
            if len(i) < 290 and len(i) > 260:
                i = str(b) + i[i.find(','):]
                f.write(i)
                b += 1
print(str(b) + ' lines copiedin test!')

