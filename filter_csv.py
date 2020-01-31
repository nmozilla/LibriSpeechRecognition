import pandas as pd
import numpy as np

#orig = pd.read_csv('../LibriSpeech/train.csv', delimiter=',')
c = 0
with open('../LibriSpeech/train.csv', 'r') as infile:
    with open('../LibriSpeech/train-small.csv','w') as f:
        f.write('idx,input,label\n')
        for i in infile.readlines():
            if len(i) < 375 and len(i) > 330:
                i = str(c) + i[i.find(','):]
                f.write(i)
                c += 1
print(str(c) + ' lines copied in train!')

d = 0

with open('../LibriSpeech/dev.csv', 'r') as infile:
    with open('../LibriSpeech/dev-small.csv','w') as f:
        f.write('idx,input,label\n')
        for i in infile.readlines():
            if len(i) < 375 and len(i) > 330:
                i = str(d) + i[i.find(','):]   
                f.write(i)
                d += 1
print(str(d) + ' lines copied in dev!')

b = 0
with open('../LibriSpeech/test.csv', 'r') as infile:
    with open('../LibriSpeech/test-small.csv','w') as f:
        f.write('idx,input,label\n')
        for i in infile.readlines():
            if len(i) < 375 and len(i) > 330:
                i = str(b) + i[i.find(','):]
                f.write(i)
                b += 1
print(str(b) + ' lines copiedin test!')

