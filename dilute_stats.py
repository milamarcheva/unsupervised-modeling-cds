import shutil
import os
import sys
import statistics
from scipy.stats import ttest_ind, ttest_rel
import matplotlib.pyplot as plt
import logging

dr = "/Users/milamarcheva/Desktop/unsupervised-modeling/outputs/25072023/dilute/"
logs = []
for root, dirs, files in os.walk(dr):
    path = root.split(os.sep)
    for file in files:
        if file == "log":
            logs.append(dr + os.path.basename(root) + '/'+ file)

print("len logs:",len(logs))

def log2dict_new(pathname):
    file1 = open(pathname, 'r')
    lines = file1.readlines()
    file1.close()
    lines = [l.strip() for l in lines]
    lines_tokenised = [l.split() for l in lines]
    logZs = []
    examples = []
    for l in lines_tokenised:
        if l[0] == 'Example':
            logZ = (l[5][:-1])
            if logZ!= "NaN":
                logZs.append([float(logZ)]) #remove the comma
                exampleno = int(l[1].split('/')[0])
                examples.append(exampleno)
    return examples, logZs


final_logZs = [log2dict_new(l)[1][-1][0] for l in logs ]

print([(logs[i].split('/')[-2].split('_')[-2], final_logZs[i]) for i in range(len(logs))])
